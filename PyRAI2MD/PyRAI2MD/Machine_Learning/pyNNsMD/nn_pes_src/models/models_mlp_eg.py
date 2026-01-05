"""
Tensorflow keras model definitions for energy and gradient.

There are two definitions: the subclassed EnergyGradientModel and a precomputed model to 
multiply with the feature derivative for training, which overwrites training/predict step.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks


from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS as hyper_model_energy_gradient
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.layers import MLP,EmptyGradient,ConstLayerNormalization,FeatureGeometric
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.loss import get_lr_metric,r2_metric


class EnergyGradientModel(ks.Model):
    """
    Subclassed tf.keras.model for energy/gradient which outputs both energy and gradient from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported for MD code.
    """
    
    def __init__(self,hyper, **kwargs):
        """
        Initialize an EnergyModel with hyperparameters.

        Args:
            hyper (dict): Hyperparamters.
            **kwargs (dict): Additional keras.model parameters.

        Returns:
            tf.keras.model.
            
        """
        super(EnergyGradientModel, self).__init__(**kwargs)
        out_dim = int( hyper['states'])
        indim = int( hyper['atoms'])
        invd_index = hyper['invd_index']
        angle_index = hyper['angle_index']
        dihyd_index = hyper['dihyd_index']
        nn_size = hyper['nn_size']
        depth = hyper['depth']
        activ = hyper['activ']
        use_reg_activ = hyper['use_reg_activ']
        use_reg_weight = hyper['use_reg_weight']
        use_reg_bias = hyper['use_reg_bias'] 
        use_dropout = hyper['use_dropout']
        dropout = hyper['dropout']
        
        use_invd_index = len(invd_index)>0 if isinstance(invd_index,list) or isinstance(invd_index,np.ndarray) else False
        use_angle_index = len(angle_index)>0 if isinstance(angle_index,list) or isinstance(angle_index,np.ndarray) else False
        use_dihyd_index = len(dihyd_index)>0 if isinstance(dihyd_index,list) or isinstance(dihyd_index,np.ndarray) else False
        
        invd_index = np.array(invd_index,dtype = np.int64) if use_invd_index else None
        angle_index = np.array(angle_index ,dtype = np.int64) if use_angle_index else None
        dihyd_index = np.array(dihyd_index,dtype = np.int64) if use_dihyd_index else None
        
        invd_shape = invd_index.shape if use_invd_index else None
        angle_shape = angle_index.shape if use_angle_index else None
        dihyd_shape = dihyd_index.shape if use_dihyd_index else None
    
        self.feat_layer = FeatureGeometric(invd_shape = invd_shape,
                                           angle_shape = angle_shape,
                                           dihyd_shape = dihyd_shape,
                                           )
        self.feat_layer.set_mol_index(invd_index, angle_index , dihyd_index)
        
        self.std_layer = ConstLayerNormalization(axis=-1,name='feat_std')
        self.mlp_layer = MLP( nn_size,
                 dense_depth = depth,
                 dense_bias = True,
                 dense_bias_last = True,
                 dense_activ = activ,
                 dense_activ_last = activ,
                 dense_activity_regularizer = use_reg_activ,
                 dense_kernel_regularizer = use_reg_weight,
                 dense_bias_regularizer = use_reg_bias,
                 dropout_use = use_dropout,
                 dropout_dropout = dropout,
                 name = 'mlp'
                 )
        self.energy_layer =  ks.layers.Dense(out_dim,name='energy',use_bias=True,activation='linear')
        
        self.build((None,indim,3))
        
    def call(self, data, training=False):
        """
        Call the model output, forward pass.

        Args:
            data (tf.tensor): Coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (list): List of tf.tensor for predicted [energy,gradient]

        """
        # Unpack the data
        x = data
        # Compute predictions
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            feat_flat = self.feat_layer(x)
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std,training=training)
            temp_e = self.energy_layer(temp_hidden)
        temp_g = tape2.batch_jacobian(temp_e, x)
        y_pred = [temp_e,temp_g]
        return y_pred



class EnergyGradientModelPrecomputed(ks.Model):
    def __init__(self,eg_atoms ,eg_states, **kwargs):
        super(EnergyGradientModelPrecomputed, self).__init__(**kwargs)
        self.eg_atoms = eg_atoms
        self.eg_states = eg_states
        self.metrics_y_gradient_std = tf.constant(np.ones((1,1,1,1)),dtype=tf.float32)
        self.metrics_y_energy_std = tf.constant(np.ones((1,1)),dtype=tf.float32)
        
    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
 
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x1)
                atpot = self(x1, training=True)[0]  # Forward pass      
            grad = tape2.batch_jacobian(atpot, x1)           
            grad = ks.backend.batch_dot(grad,x2,axes=(2,1))            
            y_pred = [atpot,grad]
            
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state([y[0]*self.metrics_y_energy_std,y[1]*self.metrics_y_gradient_std], [y_pred[0]*self.metrics_y_energy_std,y_pred[1]*self.metrics_y_gradient_std], sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape2:
            tape2.watch(x1)
            atpot = self(x1, training=False)[0]  # Forward pass 
        grad = tape2.batch_jacobian(atpot, x1)
        grad = ks.backend.batch_dot(grad,x2,axes=(2,1))            
        y_pred = [atpot,grad]
        
        self.compiled_loss(y,y_pred , regularization_losses=self.losses)
        self.compiled_metrics.update_state([y[0]*self.metrics_y_energy_std,y[1]*self.metrics_y_gradient_std], [y_pred[0]*self.metrics_y_energy_std,y_pred[1]*self.metrics_y_gradient_std], sample_weight=sample_weight)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        # Unpack the data
        x,_,_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Compute predictions
        x1 = x[0]
        x2 = x[1]
        with tf.GradientTape() as tape2:
            tape2.watch(x1)
            atpot = self(x1, training=False)[0]  # Forward pass 
        grad = tape2.batch_jacobian(atpot, x1)          
        grad = ks.backend.batch_dot(grad,x2,axes=(2,1))        
        y_pred = [atpot,grad]
        return y_pred




def create_model_energy_gradient_precomputed(hyper=hyper_model_energy_gradient['model'],
                                             learning_rate_start = 1e-3,
                                             loss_weights = [1,1]):
    """
    Full Model y = model(feat) with feat=[f,df/dx] features and its derivative to coordinates x.

    Args:
        hyper (dict, optional): Hyper dictionary. The default is hyper_model_energy_gradient['model'].
        learning_rate_start (float, optional): Initial Learning rate in compile. Defaults to 1e-3.
        loss_weights (list, optional): Weights between energy and gradient. defualt is [1,1]

    Returns:
        model (TYPE): DESCRIPTION.

    """
    out_dim = hyper['states']
    indim = int( hyper['atoms'])
    invd_index = hyper['invd_index']
    angle_index = hyper['angle_index']
    dihyd_index = hyper['dihyd_index']
    nn_size = hyper['nn_size']
    depth = hyper['depth']
    activ = hyper['activ']
    use_reg_activ = hyper['use_reg_activ']
    use_reg_weight = hyper['use_reg_weight']
    use_reg_bias = hyper['use_reg_bias'] 
    use_dropout = hyper['use_dropout']
    dropout = hyper['dropout']
    
    use_invd_index = len(invd_index)>0 if isinstance(invd_index,list) or isinstance(invd_index,np.ndarray) else False
    use_angle_index = len(angle_index)>0 if isinstance(angle_index,list) or isinstance(angle_index,np.ndarray) else False
    use_dihyd_index = len(dihyd_index)>0 if isinstance(dihyd_index,list) or isinstance(dihyd_index,np.ndarray) else False
    
    in_model_dim = 0
    if(use_invd_index==True):
        in_model_dim += len(invd_index)
    else:
        in_model_dim += int(indim*(indim-1)/2) #default is all inverse distances
    if(use_angle_index == True):
        in_model_dim += len(angle_index) 
    if(use_dihyd_index == True):
        in_model_dim += len(dihyd_index) 

    geo_input = ks.Input(shape=(in_model_dim,), dtype='float32' ,name='geo_input')
    #grad_input = ks.Input(shape=(in_model_dim,indim,3), dtype='float32' ,name='grad_input')
    
    full = ks.layers.Flatten(name='feat_flat')(geo_input)
    full = ConstLayerNormalization(name='feat_std')(full)
    full = MLP( nn_size,
             dense_depth = depth,
             dense_bias = True,
             dense_bias_last = True,
             dense_activ = activ,
             dense_activ_last = activ,
             dense_activity_regularizer = use_reg_activ,
             dense_kernel_regularizer = use_reg_weight,
             dense_bias_regularizer = use_reg_bias,
             dropout_use = use_dropout,
             dropout_dropout = dropout,
             name = 'mlp'
             )(full)
    
    energy =  ks.layers.Dense(out_dim,name='energy',use_bias=True,activation='linear')(full)
    #grads = EnergyGradient(mult_states=out_dim)([energy,geo_input])
    #force = PropagateEnergyGradient(mult_states=out_dim,name='force')([grads,grad_input])
    
    force = EmptyGradient(name='force')(geo_input)  #Will be differentiated in fit/predict/evaluate
    
    model = EnergyGradientModelPrecomputed(inputs=geo_input, outputs=[energy,force],
                        eg_atoms = indim,
                        eg_states = out_dim)
    
    #model.output_names = ['energy','force']
    #model = ks.Model(inputs=[geo_input,grad_input], outputs=[energy,grads ])
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=['mean_squared_error','mean_squared_error'],loss_weights = loss_weights,
                  metrics=['mean_absolute_error'  ,lr_metric,r2_metric])
    return model


