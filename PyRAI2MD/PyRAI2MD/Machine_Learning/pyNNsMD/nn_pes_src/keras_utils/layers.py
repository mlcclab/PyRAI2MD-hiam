"""
Keras layers for feature and model predictions around MLP.

@author: Patrick
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.keras_utils.activ import leaky_softplus,shifted_softplus

    
class ConstLayerNormalization(ks.layers.Layer):
    """
    Layer normalization with constant scaling of input.
    
    Note that this sould be replaced with keras normalization layer where trainable could be altered.
    The standardization is done via 'std' and 'mean' tf.variable and uses not very flexible broadcasting.

    """
    
    def __init__(self, axis=-1 , **kwargs):
        """
        Init the layer.

        Args:
            axis (int,list, optional): Which axis match the input on build. Defaults to -1.
            **kwargs

        """
        super(ConstLayerNormalization, self).__init__(**kwargs)          
        self.axis = axis
    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape (list): Shape of Input.

        Raises:
            TypeError: Axis argument is not valud.

        """
        super(ConstLayerNormalization, self).build(input_shape) 
        outshape = [1]*len(input_shape)
        if(isinstance(self.axis, int) == True):
            outshape[self.axis] = input_shape[self.axis]
        elif(isinstance(self.axis, list) == True or isinstance(self.axis, tuple) == True ):
            for i in self.axis:
                outshape[i] = input_shape[i]
        else:
            raise TypeError("Invalid axis argument")
        self.wmean = self.add_weight(
            'mean',
            shape=outshape,
            initializer=tf.keras.initializers.Zeros(),
            dtype=self.dtype,
            trainable=False)         
        self.wstd = self.add_weight(
            'std',
            shape=outshape,
            initializer= tf.keras.initializers.Ones(),
            dtype=self.dtype,
            trainable=False)
    def call(self, inputs):
        """
        Forward pass of the layer. Call().

        Args:
            inputs (tf.tensor): Tensor to scale.

        Returns:
            out (tf.tensor): (inputs-mean)/std

        """
        out = (inputs-self.wmean)/(self.wstd + tf.keras.backend.epsilon())
        return out 
    def get_config(self):
        """
        Config for the layer.

        Returns:
            config (dict): super.config with updated axis parameter.

        """
        config = super(ConstLayerNormalization, self).get_config()
        config.update({"axs": self.axis})
        return config 


class MLP(ks.layers.Layer):
    """
    Multilayer perceptron that consist of N dense keras layers.
    
    Last layer can be modified sperately. Hidden layers are all the same.
    """
    
    def __init__(self,
                 dense_units,
                 dense_depth = 1,
                 dense_bias = True,
                 dense_bias_last = True,
                 dense_activ = None,
                 dense_activ_last = None,
                 dense_activity_regularizer=None,
                 dense_kernel_regularizer=None,
                 dense_bias_regularizer=None,
                 dropout_use = False,
                 dropout_dropout = 0,
                 **kwargs):
        """
        Init MLP as for dense.

        Args:
            dense_units (int): Size of hidden layers.
            dense_depth (int, optional): Number of hidden layers. Defaults to 1.
            dense_bias (bool, optional): Use bias for hidden layers. Defaults to True.
            dense_bias_last (bool, optional): Bias for last layer. Defaults to True.
            dense_activ (str, optional): Activity identifier. Defaults to None.
            dense_activ_last (str, optional): Activity identifier for last layer. Defaults to None.
            dense_activity_regularizer (str, optional): Activity regularizer identifier. Defaults to None.
            dense_kernel_regularizer (str, optional): Kernel regularizer identifier. Defaults to None.
            dense_bias_regularizer (str, optional): Bias regularizer identifier. Defaults to None.
            dropout_use (bool, optional): Use dropout. Defaults to False.
            dropout_dropout (float, optional): Fraction of dropout. Defaults to 0.
            **kwargs 

        """
        super(MLP, self).__init__(**kwargs) 
        self.dense_units = dense_units
        self.dense_depth = dense_depth 
        self.dense_bias =  dense_bias  
        self.dense_bias_last = dense_bias_last 
        self.dense_activ_serialize = dense_activ
        self.dense_activ = ks.activations.deserialize(dense_activ,custom_objects={'leaky_softplus':leaky_softplus,'shifted_softplus':shifted_softplus})
        self.dense_activ_last_serialize = dense_activ_last
        self.dense_activ_last = ks.activations.deserialize(dense_activ_last,custom_objects={'leaky_softplus':leaky_softplus,'shifted_softplus':shifted_softplus})
        self.dense_activity_regularizer = ks.regularizers.get(dense_activity_regularizer)
        self.dense_kernel_regularizer = ks.regularizers.get(dense_kernel_regularizer)
        self.dense_bias_regularizer = ks.regularizers.get(dense_bias_regularizer)
        self.dropout_use = dropout_use
        self.dropout_dropout = dropout_dropout
        
        self.mlp_dense_activ = [ks.layers.Dense(
                                self.dense_units,
                                use_bias=self.dense_bias,
                                activation=self.dense_activ,
                                name=self.name+'_dense_'+str(i),
                                activity_regularizer= self.dense_activity_regularizer,
                                kernel_regularizer=self.dense_kernel_regularizer,
                                bias_regularizer=self.dense_bias_regularizer
                                ) for i in range(self.dense_depth-1)]
        self.mlp_dense_last =  ks.layers.Dense(
                                self.dense_units,
                                use_bias=self.dense_bias_last,
                                activation=self.dense_activ_last,
                                name= self.name + '_last',
                                activity_regularizer= self.dense_activity_regularizer,
                                kernel_regularizer=self.dense_kernel_regularizer,
                                bias_regularizer=self.dense_bias_regularizer
                                )
        if(self.dropout_use == True):
            self.mlp_dropout =  ks.layers.Dropout(self.dropout_dropout,name=self.name + '_dropout')
    def build(self, input_shape):
        """
        Build layer.

        Args:
            input_shape (list): Input shape.

        """
        super(MLP, self).build(input_shape)          
    def call(self, inputs,training=False):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Input tensor of shape (...,N).
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            out (tf.tensor): Last activity.

        """
        x = inputs
        for i in range(self.dense_depth-1):
            x = self.mlp_dense_activ[i](x)
            if(self.dropout_use == True):
                x = self.mlp_dropout(x,training=training)
        x = self.mlp_dense_last(x)
        out = x
        return out
    def get_config(self):
        """
        Update config.

        Returns:
            config (dict): Base class config plus MLP info.

        """
        config = super(MLP, self).get_config()
        config.update({"dense_units": self.dense_units,
                       'dense_depth': self.dense_depth,
                       'dense_bias': self.dense_bias,
                       'dense_bias_last': self.dense_bias_last,
                       'dense_activ' : self.dense_activ_serialize,
                       'dense_activ_last' : self.dense_activ_last_serialize,
                       'dense_activity_regularizer': ks.regularizers.serialize(self.dense_activity_regularizer),
                       'dense_kernel_regularizer': ks.regularizers.serialize(self.dense_kernel_regularizer),
                       'dense_bias_regularizer': ks.regularizers.serialize(self.dense_bias_regularizer),
                       'dropout_use': self.dropout_use,
                       'dropout_dropout': self.dropout_dropout
                       })
        return config


class InverseDistance(ks.layers.Layer):
    def __init__(self , **kwargs):
        super(InverseDistance, self).__init__(**kwargs)  
        #self.dinv_mean = dinv_mean
        #self.dinv_std = dinv_std
    def build(self, input_shape):
        super(InverseDistance, self).build(input_shape)          
    def call(self, inputs):
        coords = inputs #(batch,N,3)
        #Compute square dinstance matrix
        ins_int = ks.backend.int_shape(coords)
        ins = ks.backend.shape(coords)
        a = ks.backend.expand_dims(coords ,axis = 1)
        b = ks.backend.expand_dims(coords ,axis = 2)
        c = b-a #(batch,N,N,3)
        d = ks.backend.sum(ks.backend.square(c),axis = -1) #squared distance without sqrt for derivative
        #Compute Mask for lower tri
        ind1 = ks.backend.expand_dims(ks.backend.arange(0,ins_int[1]),axis=1)
        ind2 = ks.backend.expand_dims(ks.backend.arange(0,ins_int[1]),axis=0)
        mask = ks.backend.less(ind1,ind2)
        mask = ks.backend.expand_dims(mask,axis=0)
        mask = ks.backend.tile(mask,(ins[0],1,1)) #(batch,N,N)
        #Apply Mask and reshape 
        d = d[mask]
        d = ks.backend.reshape(d,(ins[0],ins_int[1]*(ins_int[1]-1)//2)) # Not pretty
        d = ks.backend.sqrt(d) #Now the sqrt is okay
        out = 1/d #Now inverse should also be okay
        #out = (out - self.dinv_mean )/self.dinv_std #standardize with fixed values.
        return out 


class InverseDistanceIndexed(ks.layers.Layer):
    """
    Compute inverse distances from coordinates.
    
    The index-list of atoms to compute distances from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """
    
    def __init__(self, invd_shape, **kwargs):
        """
        Init the layer. The index list is initialized to zero.

        Args:
            invd_shape (list): Shape of the index piar list without batch dimension (N,2).
            **kwargs.
            
        """
        super(InverseDistanceIndexed, self).__init__(**kwargs)  
        self.invd_shape = invd_shape

        self.invd_list = self.add_weight('invd_list',
                                        shape=invd_shape,
                                        initializer=tf.keras.initializers.Zeros(),
                                        dtype='int64',
                                        trainable=False)         
    def build(self, input_shape):
        """
        Build model. Index list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(InverseDistanceIndexed, self).build(input_shape)          
    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinate input as (batch,N,3).

        Returns:
            angs_rad (tf.tensor): Flatten list of angles from index.

        """
        cordbatch  = inputs
        invdbatch  = tf.repeat(ks.backend.expand_dims(self.invd_list,axis=0) , ks.backend.shape(cordbatch)[0], axis=0)
        vcords1 = tf.gather(cordbatch, invdbatch[:,:,0],axis=1,batch_dims=1)
        vcords2 = tf.gather(cordbatch, invdbatch[:,:,1],axis=1,batch_dims=1)
        vec=vcords2-vcords1
        norm_vec = ks.backend.sqrt(ks.backend.sum(vec*vec,axis=-1))
        invd_out = tf.math.divide_no_nan(tf.ones_like(norm_vec),norm_vec)
        return invd_out
    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle invd shape.

        """
        config = super(InverseDistanceIndexed, self).get_config()
        config.update({"invd_shape": self.invd_shape})
        return config



class Angles(ks.layers.Layer):
    """
    Compute angles from coordinates.
    
    The index-list of atoms to compute angles from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """
    
    def __init__(self, angle_shape, **kwargs):
        """
        Init the layer. The angle list is initialized to zero.

        Args:
            angle_shape (list): Shape of the angle list without batch dimension (N,3).
            **kwargs.
            
        """
        super(Angles, self).__init__(**kwargs)  
        #self.angle_list = angle_list
        #self.angle_list_tf = tf.constant(np.array(angle_list))
        self.angle_list = self.add_weight('angle_list',
                                        shape=angle_shape,
                                        initializer=tf.keras.initializers.Zeros(),
                                        dtype='int64',
                                        trainable=False)         
    def build(self, input_shape):
        """
        Build model. Angle list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(Angles, self).build(input_shape)          
    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinate input as (batch,N,3).

        Returns:
            angs_rad (tf.tensor): Flatten list of angles from index.

        """
        cordbatch  = inputs
        angbatch  = tf.repeat(ks.backend.expand_dims(self.angle_list,axis=0) , ks.backend.shape(cordbatch)[0], axis=0)
        vcords1 = tf.gather(cordbatch, angbatch[:,:,1],axis=1,batch_dims=1)
        vcords2a = tf.gather(cordbatch, angbatch[:,:,0],axis=1,batch_dims=1)
        vcords2b = tf.gather(cordbatch, angbatch[:,:,2],axis=1,batch_dims=1)
        vec1=vcords2a-vcords1
        vec2=vcords2b-vcords1
        norm_vec1 = ks.backend.sqrt(ks.backend.sum(vec1*vec1,axis=-1))
        norm_vec2 = ks.backend.sqrt(ks.backend.sum(vec2*vec2,axis=-1))
        angle_cos = ks.backend.sum(vec1*vec2,axis=-1)/ norm_vec1 /norm_vec2
        angs_rad = tf.math.acos(angle_cos)
        return angs_rad
    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle index shape.

        """
        config = super(Angles, self).get_config()
        config.update({"angle_shape": self.angle_shape})
        return config


class Dihydral(ks.layers.Layer):
    """
    Compute dihydral angles from coordinates.
    
    The index-list of atoms to compute angles from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """
    
    def __init__(self ,angle_shape, **kwargs):
        """
        Init the layer. The angle list is initialized to zero.

        Args:
            angle_shape (list): Shape of the angle list without batch dimension of (N,4).
            **kwargs

        """
        super(Dihydral, self).__init__(**kwargs)  
        #self.angle_list = angle_list
        #self.angle_list_tf = tf.constant(np.array(angle_list))
        self.angle_list = self.add_weight('angle_list',
                                shape=angle_shape,
                                initializer=tf.keras.initializers.Zeros(),
                                dtype='int64',
                                trainable=False)        
    def build(self, input_shape):
        """
        Build model. Angle list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(Dihydral, self).build(input_shape)          
    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinates of shape (batch, N,3).

        Returns:
            angs_rad (tf.tensor): Dihydral angles from index list and coordinates of shape (batch,M).

        """
        #implementation from
        #https://en.wikipedia.org/wiki/Dihedral_angle
        cordbatch = inputs
        indexbatch  = tf.repeat(ks.backend.expand_dims(self.angle_list,axis=0) , ks.backend.shape(cordbatch)[0], axis=0)
        p1 = tf.gather(cordbatch, indexbatch[:,:,0],axis=1,batch_dims=1)
        p2 = tf.gather(cordbatch, indexbatch[:,:,1],axis=1,batch_dims=1)
        p3 = tf.gather(cordbatch, indexbatch[:,:,2],axis=1,batch_dims=1)
        p4 = tf.gather(cordbatch, indexbatch[:,:,3],axis=1,batch_dims=1)
        b1 = p1-p2  
        b2 = p2-p3
        b3 = p4-p3
        arg1 = ks.backend.sum(b2*tf.linalg.cross(tf.linalg.cross(b3,b2),tf.linalg.cross(b1,b2)),axis=-1)
        arg2 = ks.backend.sqrt(ks.backend.sum(b2*b2,axis=-1))*ks.backend.sum(tf.linalg.cross(b1,b2)*tf.linalg.cross(b3,b2),axis=-1)
        angs_rad = tf.math.atan2(arg1,arg2) 
        return angs_rad
    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle index shape.

        """
        config = super(Dihydral, self).get_config()
        config.update({"angle_shape": self.angle_shape})
        return config


class FeatureGeometric(ks.layers.Layer):
    """
    Feautre representation consisting of inverse distances, angles and dihydral angles.
    
    Uses InverseDistance, Angle, Dihydral layer definition if input index is not empty.
    
    """
    
    def __init__(self ,
                 invd_shape = None,
                 angle_shape = None,
                 dihyd_shape = None,
                 **kwargs):
        """
        Init of the layer.

        Args:
            invd_shape (list, optional): Index-Shape of atoms to calculate inverse distances. Defaults to None.
            angle_shape (list, optional): Index-Shape of atoms to calculate angles between. Defaults to None.
            dihyd_shape (list, optional): Index-Shape of atoms to calculate dihyd between. Defaults to None.
            **kwargs

        """
        super(FeatureGeometric, self).__init__(**kwargs)
        #Inverse distances are always taken all for the moment
        self.use_invdist = invd_shape is not None
        self.invd_shape = invd_shape
        self.use_bond_angles = angle_shape is not None
        self.angle_shape = angle_shape 
        self.use_dihyd_angles = dihyd_shape is not None
        self.dihyd_shape = dihyd_shape

        if(self.use_invdist==True):        
            self.invd_layer = InverseDistanceIndexed(invd_shape)
        else:
            self.invd_layer = InverseDistance() #default always
        if(self.use_bond_angles==True):
            self.ang_layer = Angles(angle_shape=angle_shape)
            self.concat_ang = ks.layers.Concatenate(axis=-1)
        if(self.use_dihyd_angles==True):
            self.dih_layer = Dihydral(angle_shape=dihyd_shape)
            self.concat_dih = ks.layers.Concatenate(axis=-1)
        self.flat_layer = ks.layers.Flatten(name='feat_flat')
    def build(self, input_shape):
        """
        Build model. Passes to base class.

        Args:
            input_shape (list): Input shape.

        """
        super(FeatureGeometric, self).build(input_shape)
    def call(self, inputs):
        """
        Forward pass of the layer. Call().

        Args:
            inputs (tf.tensor): Coordinates of shape (batch,N,3).

        Returns:
            out (tf.tensor): Feature description of shape (batch,M).

        """
        x = inputs
        
        feat = self.invd_layer(x)
        if(self.use_bond_angles==True):
            angs = self.ang_layer(x)
            feat = self.concat_ang([feat,angs])
        if(self.use_dihyd_angles==True):
            dih = self.dih_layer(x)
            feat = self.concat_dih([feat,dih])
    
        feat_flat = self.flat_layer(feat)
        out = feat_flat
        return out 
    def set_mol_index(self,invd_index,angle_index,dihyd_index):
        """
        Set weights for atomic index for distance and angles.

        Args:
            invd_index (np.array): Index for inverse distances. Shape (N,2)
            angle_index (np.array): Index for angles. Shape (N,3).
            dihyd_index (np.array):Index for dihed angles. Shape (N,4).

        """
        if(self.use_invdist==True):
            self.invd_layer.set_weights([invd_index])
        if(self.use_dihyd_angles == True):
            self.dih_layer.set_weights([dihyd_index])
        if(self.use_bond_angles == True):
            self.ang_layer.set_weights([angle_index])
    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus index info.

        """
        config = super(FeatureGeometric, self).get_config()
        config.update({"invd_shape" : self.invd_shape,
                       "angle_shape" : self.angle_shape,
                       "dihyd_shape" : self.dihyd_shape
                       })
        return config
    

class EnergyGradient(ks.layers.Layer):
    def __init__(self, mult_states = 1, **kwargs):
        super(EnergyGradient, self).__init__(**kwargs)          
        self.mult_states = mult_states
    def build(self, input_shape):
        super(EnergyGradient, self).build(input_shape)          
    def call(self, inputs):
        energy,coords = inputs
        out = [ks.backend.expand_dims(ks.backend.gradients(energy[:,i],coords)[0],axis=1) for i in range(self.mult_states)]
        out = ks.backend.concatenate(out,axis=1)   
        return out
    def get_config(self):
        config = super(EnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config


class NACGradient(ks.layers.Layer):
    def __init__(self, mult_states = 1, atoms = 1, **kwargs):
        super(NACGradient, self).__init__(**kwargs)          
        self.mult_states = mult_states
        self.atoms=atoms
    def build(self, input_shape):
        super(NACGradient, self).build(input_shape)          
    def call(self, inputs):
        energy,coords = inputs
        out = ks.backend.concatenate([ks.backend.expand_dims(ks.backend.gradients(energy[:,i],coords)[0],axis=1) for i in range(self.mult_states*self.atoms)],axis=1)
        out = ks.backend.reshape(out,(ks.backend.shape(coords)[0],self.mult_states,self.atoms,self.atoms,3))
        out = ks.backend.concatenate([ks.backend.expand_dims(out[:,:,i,i,:],axis=2) for i in range(self.atoms)],axis=2)
        return out
    def get_config(self):
        config = super(NACGradient, self).get_config()
        config.update({"mult_states": self.mult_states,'atoms': self.atoms})
        return config 
    
    
class EmptyGradient(ks.layers.Layer):
    def __init__(self, mult_states = 1, atoms = 1, **kwargs):
        super(EmptyGradient, self).__init__(**kwargs)          
        self.mult_states = mult_states
        self.atoms=atoms
    def build(self, input_shape):
        super(EmptyGradient, self).build(input_shape)          
    def call(self, inputs):
        pot = inputs
        out = tf.zeros((ks.backend.shape(pot)[0],self.mult_states,self.atoms,3))
        return out
    def get_config(self):
        config = super(EmptyGradient, self).get_config()
        config.update({"mult_states": self.mult_states,'atoms': self.atoms})
        return config 
    

class PropagateEnergyGradient(ks.layers.Layer):
    def __init__(self,mult_states = 1 ,**kwargs):
        super(PropagateEnergyGradient, self).__init__(**kwargs) 
        self.mult_states = mult_states         
    def build(self, input_shape):
        super(PropagateEnergyGradient, self).build(input_shape)          
    def call(self, inputs):
        grads,grads2 = inputs
        out = ks.backend.batch_dot(grads,grads2,axes=(2,1))
        return out
    def get_config(self):
        config = super(PropagateEnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config 


class PropagateNACGradient(ks.layers.Layer):
    def __init__(self,mult_states = 1,atoms=1 ,**kwargs):
        super(PropagateNACGradient, self).__init__(**kwargs) 
        self.mult_states = mult_states 
        self.atoms = atoms        
    def build(self, input_shape):
        super(PropagateNACGradient, self).build(input_shape)          
    def call(self, inputs):
        grads,grads2 = inputs
        out = ks.backend.batch_dot(grads,grads2,axes=(3,1))
        out = ks.backend.concatenate([ks.backend.expand_dims(out[:,:,i,i,:],axis=2) for i in range(self.atoms)],axis=2)
        return out
    def get_config(self):
        config = super(PropagateNACGradient, self).get_config()
        config.update({"mult_states": self.mult_states,'atoms': self.atoms})
        return config 

class PropagateNACGradient2(ks.layers.Layer):
    def __init__(self,axis=(2,1),**kwargs):
        super(PropagateNACGradient2, self).__init__(**kwargs) 
        self.axis = axis
    def build(self, input_shape):
        super(PropagateNACGradient2, self).build(input_shape)          
    def call(self, inputs):
        grads,grads2 = inputs
        out = ks.backend.batch_dot(grads,grads2,axes=self.axis)        
        return out
    def get_config(self):
        config = super(PropagateNACGradient2, self).get_config()
        config.update({"axis": self.axis})
        return config 