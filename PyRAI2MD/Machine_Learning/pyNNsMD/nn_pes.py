"""
Main class for neural network (NN) container to provide multiple NN instances.

Enables uncertainty estimate as well as training and prediction of tf.keras models
for potentials plus gradients, couplings and further models. 
The python class is supposed to allow parallel training and 
further operations like resampling and hyperparameter optimization. 
"""

#import logging
import os
import sys
import numpy as np
import tensorflow as tf


from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.modeltype import _get_model_by_type
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.hyper import _save_hyp,_load_hyp,_get_default_hyperparameters_by_modeltype
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.fit import _fit_model_by_modeltype
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.data import model_save_data_to_folder,model_make_random_shuffle,model_merge_data_in_chunks,index_make_random_shuffle
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.scaler import get_default_scaler
from PyRAI2MD.Machine_Learning.pyNNsMD.nn_pes_src.predict import _predict_uncertainty,_call_convert_x_to_tensor,_call_convert_y_to_numpy


class NeuralNetPes:
    """
    Main class NeuralNetPes(directory) that keeps multiple keras models and manages training and prediction.
    
    The individual model types are further stored to file in the directory specified in __init__(directory). 
    Each model you create within a NeuralNetPes is referenced by a dictionary.
    The information like predictions and hyperparameters are also passed in form of python dictionaries. 
    See the default parameters in nn_pes_src.hypers for the scope of all parameters and their explanation. 
    
    Example:
        nn = NeuralNetPes("mol2")
        hyper_energy = {'general' : {'model_type':'mlp_eg'} , 'model' : {'atoms':12} , 'training' : {}} 
        hyper_nac = {'general': {'model_type':'mlp_nac'} , 'model': {'atoms':12} , 'training' : {}}
        nn.create({'energy': hyper_energy , 'nac' : hyper_nac})
    
    """
    
    def __init__(self,directory: str,mult_nn = 2):
        """
        Initilialize empty NeuralNetPes instance.

        Args:
            directory (str): Directory where models, hyperparameter, logs and fitresults are stored.
            mult_nn (TYPE, optional): Number of NN instances to create for error estimate. The default is 2.

        Returns:
            NueralNetPes instance.
        """
        self._models_implemented = ['mlp_eg', 'mlp_nac','mlp_nac2','mlp_e']

        #self.logger = logging.getLogger(type(self).__name__)
        print("Info: Operating System: ",sys.platform)
        print("Info: Tested for tf-gpu= 2.3 This tf version: ",tf.__version__)        
        print("Info: Models implemented:" , self._models_implemented)
        
        # Private memebers
        self._models = {}
        self._models_hyper = {}
        self._models_scaler = {}
        
        self._directory = directory
        self._addNN = mult_nn
        
        self._last_shuffle = None
        
    
    def _merge_hyper(self,dictold, dictnew,exclude_category = []):
        """
        Merge hyperparameter dictionaries on the first two levels.

        Args:
            dictold (dict): Existing or eg. default dict.
            dictnew (dict): Dict to update.
            exclude_category (dict, optional): Exclude a category not to update. Defaults to [].

        Returns:
            temp (dict): Updated dict.

        """
        temp = {}
        temp.update(dictold)
        for hkey in dictnew.keys():
            is_new_category = False
            if(hkey not in temp):
                print("Warning: Unknown category:", hkey, ". For new category check necessary hyper parameters, warnings for dict-keys are suppressed.")
                is_new_category = True
                temp[hkey] = {}
            if(hkey in exclude_category):
                print("Error: Can not update specific category",hkey)
            else:
                for hhkey in dictnew[hkey].keys():
                    if(hhkey not in temp[hkey] and is_new_category == False):
                        print("Warning: Unknown key:", hhkey , "in", hkey)
                    temp[hkey][hhkey] = dictnew[hkey][hhkey]
        return temp

    
    def _create_models(self,key,h_dict):
        #Check if hpyer is a list of dict or a single dict
        if(isinstance(h_dict, dict)):
            model_type = h_dict['general']['model_type']
            #Make a list with the same hyp
            hyp = [h_dict for x in range(self._addNN)]
        elif(isinstance(h_dict, list)):
            if(len(h_dict) != self._addNN):
                print(f"Error: Error in hyp for number NNs for {key}")
            model_type = h_dict[0]['general']['model_type']
            for x in h_dict:
                if(x['general']['model_type'] != model_type):
                    print(f"Error: Inconsistent Input for {key}")
            #Accept list of hyperdicts
            hyp = h_dict
        else:
            raise TypeError(f"Unknwon Input tpye of hyper for {key}")
        
        # Create the correct model with hyper
        models = {}
        models[key] = []
        scaler = {}
        scaler[key] = []
        for i in range(self._addNN):
            #Fill missing hyper
            temp = self._merge_hyper(_get_default_hyperparameters_by_modeltype(model_type),hyp[i])              
            hyp[i] = temp
            models[key].append(_get_model_by_type(model_type,hyper=temp['model']))
            scaler[key].append(get_default_scaler(model_type))
            
        return models,hyp,scaler
    
    
    def create(self,hyp_dict):
        """
        Initialize and build a model. Missing hyperparameter are filled from default.
        
        The model name can be freely chosen. Thy model type is determined in hyper_dict.

        Args:
            hyp_dict (dict): Dictionary with hyper-parameter. {'model' : hyper_dict, 'model2' : hyper_dict2 ....}
                             Missing hyperparameter in hyper_dict are filled up from default, see nn_pes_src.hypers for complete set.

        Returns:
            list: created models.

        """
        for key, value in hyp_dict.items():
            mod,hyp,sc = self._create_models(key,value)
            self._models.update(mod)
            self._models_hyper[key] = hyp
            self._models_scaler.update(sc)

        return self._models
   
    
    def update(self,hyp_dict):
        """
        Update hyper parameters if possible.

        Args:
            hyp_dict (dict): Dictionary with hyper-parameter. {'model' : hyper_dict, 'model2' : hyper_dict2 , ...} to update.
                             Note: model parameters will not be updated.

        Raises:
            TypeError: If input type is not compatible with number of models.

        Returns:
            None.

        """
        for key, value in hyp_dict.items():
            if(key in self._models_hyper.keys()):
                if(isinstance(value, dict)):
                    value = [value for i in range(self._addNN)]
                elif(len(value) != self._addNN):
                    print(f"Error: Error in hyp for number NNs for {key}")
                else:
                    print(f"Error: Unknwon Input tpye of hyper for {key}")
                    raise TypeError(f"Unknwon Input tpye of hyper for {key}")
                for i in range(self._addNN):
                    self._models_hyper[key][i] = self._merge_hyper(self._models_hyper[key][i],
                                                                   value[i],
                                                                   exclude_category=['model'])
    
    
    def _save(self,directory,name):
        # Check if model name can be saved
        if(name not in self._models):
            raise TypeError("Cannot save model before init.")
            
        # Folder to store model in
        filename = os.path.abspath(os.path.join(directory,name))
        os.makedirs(filename,exist_ok=True)
        
        #Store weights and hyper
        for i,x in enumerate(self._models[name]):
            x.save_weights(os.path.join(filename,'weights'+'_v%i'%i+'.h5'))
        for i,x in enumerate(self._models_hyper[name]):
            _save_hyp(x,os.path.join(filename,'hyper'+'_v%i'%i+".json"))
        for i,x in enumerate(self._models_scaler[name]):   
            x.save(os.path.join(filename,'scaler'+'_v%i'%i+".json"))
        
        return filename
        
    
    def save(self,model_name=None):
        """
        Save a model weights and hyperparameter into class folder with a certain name.
        
        The model itself is not saved, use export to store the model itself.
        Thes same holds for load. Here the model is recreated from hyperparameters and weights.     

        Args:
            model_name (str, optional): Name of the Model to save. The default is None, which means save all

        Returns:
            out_dirs (list): Saved directories.

        """
        dirname = self._directory
        directory = os.path.abspath(dirname)
        os.makedirs(directory, exist_ok=True)
        
        # Safe model_name 
        out_dirs = []
        if(isinstance(model_name, str)):
            out_dirs.append(self._save(directory,model_name))
            
        elif(isinstance(model_name, list)):
            for name in model_name:
                out_dirs.append(self._save(directory,name))
        
        #Default just save all
        else:    
            for name in self._models.keys():
                out_dirs.append(self._save(directory,name))
        
        return out_dirs


    def _export(self,directory,name):
        # Check if model name can be saved
        if(name not in self._models):
            raise TypeError("Cannot save model before init.")
            
        # Folder to store model in
        filename = os.path.abspath(os.path.join(directory,name))
        os.makedirs(filename,exist_ok=True)
        
        #Store model
        for i,x in enumerate(self._models[name]):
            x.save(os.path.join(filename,'SavedModel'+'_v%i'%i))
            # TFLite converison does not work atm
            # converter = tf.lite.TFLiteConverter.from_keras_model(x)
            # tflite_model = converter.convert()
            # # Save the model.
            # with open(os.path.join(filename,'LiteModel'+'_v%i'%i+'.tflite'), 'wb') as f:
            #   f.write(tflite_model)
        
        return filename
        


    def export(self,model_name=None):
        """
        Save SavedModel file into class folder with a certain name.

        Args:
            model_name (str, optional): Name of the Model to save. The default is None, which means save all

        Returns:
            out_dirs (list): Saved directories.

        """
        dirname = self._directory
        directory = os.path.abspath(dirname)
        os.makedirs(directory, exist_ok=True)
        
        # Safe model_name 
        out_dirs = []
        if(isinstance(model_name, str)):
            out_dirs.append(self._export(directory,model_name))
            
        elif(isinstance(model_name, list)):
            for name in model_name:
                out_dirs.append(self._export(directory,name))
        
        #Default just save all
        else:    
            for name in self._models.keys():
                out_dirs.append(self._export(directory,name))
        
        return out_dirs
   
    
    def _load(self, folder, model_name):
        fname = os.path.join(folder,model_name)
        # Check if folder exists
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Cannot find model directory {fname}")
        
        #Load  Hyperparameters
        hyp = []
        for i in range(self._addNN):
            hyp.append(_load_hyp(os.path.join(fname,'hyper'+'_v%i'%i+".json")))
        
        #Create Model
        self.create({model_name : hyp})
        
        #Load weights
        for i in range(self._addNN):
            self._models[model_name][i].load_weights(os.path.join(fname,'weights'+'_v%i'%i+'.h5'))
            print("Info: Imported weights for: %s"%(model_name+'_v%i'%i))
        
        #Load scaler  
        for i in range(self._addNN):
            self._models_scaler[model_name][i].load(os.path.join(fname,'scaler'+'_v%i'%i+".json"))
            print("Info: Imported scaling for: %s"%(model_name+'_v%i'%i))
                     
    
    def load(self, model_name=None):
        """
        Load a model from weights and hyperparamter that are stored in class folder.
        
        The tensorflow.keras.model is not loaded itself but created new from hyperparameters.

        Args:
            model_name (str,list, optional): Model names on file. The default is None.

        Raises:
            FileNotFoundError: If Directory not found.

        Returns:
            list: Loaded Models.

        """
        if not os.path.exists(self._directory):
            raise FileNotFoundError("Cannot find class directory")
        directory = os.path.abspath(self._directory)
        
        # Load model_name 
        if(isinstance(model_name, str)):
            self._load(directory,model_name)
            
        elif(isinstance(model_name, list)):
            for name in model_name:
                self._load(directory,name)
        
        #Default just save all
        else:
            savemod_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(self._directory, f))]    
            for name in savemod_list:
                self._load(directory,name)
        
        print("Debug: loaded all models.")
        return self._models
    
      
    def _fit_models(self, target_model,x, y,gpu,proc_async,fitmode,random_shuffle=False):
        # Pick modeltype from first hyper
        model_type = self._models_hyper[target_model][0]['general']['model_type']
        # modelfolder
        mod_dir = os.path.join(os.path.abspath(self._directory),target_model)
        #Save data, will be made model specific if necessary in the future
        model_save_data_to_folder(model_type,x, y,target_model,mod_dir,random_shuffle)
        #Start proc per NN
        proclist = []
        for i in range(self._addNN):
            proclist.append(_fit_model_by_modeltype(model_type,proc_async, i, mod_dir, gpu[i], fitmode))
     
        print(f"Debug: successfully started training for models {target_model}")
        return proclist
                 
 
    def _read_fit_error(self,target_model):
        # modelfolder
        outdir = os.path.join(os.path.abspath(self._directory),target_model)
        fit_error = []
        try:
            for i in range(self._addNN):
                error_val = np.load(os.path.join(outdir,"fiterr_valid" +'_v%i'%i+ ".npy"))
                fit_error.append(error_val)
        except:
            print(f"Error: Can not find fit error output {target_model}. Fit may not have run correctly!")
        
        return fit_error
    
    
    def fit(self, x, y , gpu_dist = {}, proc_async = True, fitmode= "training", random_shuffle = False):
        """
        Fit NN to data. Model weights and hyper parameters are always saved to file before fit.
        
        The fit routine calls training scripts on the datafolder with parallel runtime.
        The type of execution is found in nn_pes_src.fit with the training nn_pes_src.training_ scripts.
        
        Args:
            x (np.array,list,dict): X-values, e.g. Coordinates in Angstroem of shape (batch,Atoms,3)
                                    Or list of np.arrays or dataforms that can be pickled.
                                    If different models require also different x-values, provide a dict matching y.
            y (dict):   dictionary of y values for each model. 
                        Units of gradients and x-values, i.e. coordinates must match.
            gpu_dist (dict, optional):  Dictionary with same modelname and list of GPUs for each NN. Default is {}
                                        Example {'nac' : [0,0] } both NNs for NAC on GPU:0
            proc_async (bool): Try to run parallel. Default is true.
            fitmode (str):  Whether to do 'training' or 'retraining' the existing model in hyperparameter category. Default is 'training'.  
                            In principle every reasonable category can be created in hyperparameters.
            random_shuffle (bool): Whether to shuffle data before fitting. Default is False.  
            
        Returns:
            ferr (dict): Fitting Error.

        """
        #List of models
        models_available = sorted(list(self._models.keys()))
        models_to_train = sorted(list(y.keys()))
        
        #Check if model can be identified
        for modtofit in models_to_train:
            if(modtofit not in models_available):
                raise TypeError(f"Cannot train on data: {models_to_train} does not match models {models_available}!")
        
        #Check GPU Assignment and default to -1
        gpu_dict_clean = {}
        for modtofit in models_to_train:
            if(modtofit in gpu_dist):
                gpu_dict_clean[modtofit] = gpu_dist[modtofit]
            else:
                gpu_dict_clean[modtofit] = [-1 for i in range(self._addNN)] 
        
        #Fitting
        proclist = []
        for target_model, ydata in y.items():
            #Save model here with hyper !!!!
            self.save(target_model)
            print(f"Debug: starting training model {target_model}")
            if(isinstance(x,dict) == True):
                x_model = x[target_model]
            else:
                x_model = x
            proclist += self._fit_models(target_model,x_model, ydata,gpu_dict_clean[target_model],proc_async,fitmode,random_shuffle)
        
        #Wait for fits
        if(proc_async == True):
            print("Fits submitted, waiting...")
            #Wait for models to finish
            for proc in proclist:
                proc.wait()    
        
        #Look for fiterror in folder
        print("Seraching Folder for fitresult...")
        self.load(models_to_train)
        fit_error = {}
        for target_model in y.keys():
            fit_error[target_model] = self._read_fit_error(target_model)
            

        return fit_error


    def _predict_model_list(self,x_list,model_list,batch_size_list):
        out = [model_list[i].predict(x_list[i],batch_size=batch_size_list[i]) for i in range(len(model_list))]
        return out

    
    def _predict_models(self,name,x): 
        #Check type with first hyper
        out = []
        model_type = self._models_hyper[name][0]['general']['model_type']
        x_scaled = [self._models_scaler[name][i].scale_x(x) for i in range(self._addNN)]
        temp =  self._predict_model_list(x_scaled ,self._models[name],[self._models_hyper[name][i]['predict']['batch_size_predict'] for i in range(self._addNN)])
        out = [self._models_scaler[name][i].rescale_y(temp[i]) for i in range(self._addNN)]
        return _predict_uncertainty(model_type,out)
        
    
    def predict(self, x):
        """
        Prediction for all models available. Prediction is slower but works on large data.

        Args:
            x (np.array,list, dict):    Coordinates in Angstroem of shape (batch,Atoms,3)
                                        Or a suitable list of geometric input.
                                        If models require different x please provide dict matching model name.

        Returns:
            result (dict): All model predictions: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.
            error (dict): Error estimate for each value: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.

        """
        result = {}
        error = {}
        for name in self._models.keys():
            if(isinstance(x,dict) == True):
                x_model = x[name]
            else:
                x_model = x
            temp = self._predict_models(name,x_model)
            result[name] = temp[0]
            error[name] = temp[1]
        
        return result,error
    
    
    @tf.function
    def _call_model_list(self,x_list,model_list):
        out = [model_list[i](x_list[i],training=False) for i in range(len(model_list))]
        return out

    
    def _call_models(self,name,x): 
        #Check type with first hyper
        out = []
        model_type = self._models_hyper[name][0]['general']['model_type']
        x_scaled = [self._models_scaler[name][i].scale_x(x) for i in range(self._addNN)]
        x_res = [_call_convert_x_to_tensor(model_type,xs) for xs in x_scaled]
        temp =  self._call_model_list(x_res,self._models[name])
        temp = [_call_convert_y_to_numpy(model_type,xout) for xout in temp]
        out = [self._models_scaler[name][i].rescale_y(temp[i]) for i in range(self._addNN)]
        return _predict_uncertainty(model_type,out)

    
    
    def call(self,x):
        """
        Faster prediction without looping batches. Requires single small batch (batch, Atoms,3) that fit into memory.

        Args:
            x (np.array):   Coordinates in Angstroem of shape (batch,Atoms,3)
                            Or a suitable list of geometric input.
                            If models require different x please provide dict matching model name.

        Returns:
            result (dict): All model predictions: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.
            error (dict): Error estimate for each value: {'energy_gradient' : [np.aray,np.array] , 'nac' : np.array , ..}.

        """
        result = {}
        error = {}
        for name in self._models.keys():
            if(isinstance(x,dict) == True):
                x_model = x[name]
            else:
                x_model = x
            temp = self._call_models(name,x_model)
            result[name] = temp[0]
            error[name] = temp[1]
        
        
        return result,error

    
    def shuffle(self,x,y,dat_size):
        """
        Shuffle datalist consistently, i.e. each data in x,y in the same way.

        Args:
            x (np.array,list,dict): X-values, e.g. Coordinates in Angstroem of shape (batch,Atoms,3)
                                    Or list of np.arrays or dataforms that can be pickled.
                                    If different models require also different x-values, provide a dict matching y.
            y (dict):   dictionary of y values for each model.
            dat_size (int): Size of the Dataset. Must match the data.

        Returns:
            shuffle_ind (np.array): Index assignment for the shuffle for x,y etc.
            x_dict (dict): Shuffled list of the x data.
            y_dict (dict): Shuffled list of the y data.

        """
        x_dict = {}
        y_dict = {}
        shuffle_ind = index_make_random_shuffle(np.arange(dat_size))
        models_available = sorted(list(self._models.keys()))
        for name in list(y.keys()):
            if(name not in models_available):
                print("Error: model not available:",name)
            else:
                model_type = self._models_hyper[name][0]['general']['model_type']
                my = y[name]
                if(isinstance(x,dict) == True):
                    mx = x[name]
                else:
                    mx = x
                temp_x,temp_y = model_make_random_shuffle(model_type,mx,my,shuffle_ind)
                x_dict.update({name:temp_x})
                y_dict.update({name:temp_y})
            
        self._last_shuffle = shuffle_ind
        return shuffle_ind, x_dict, y_dict
    


    def merge(self,x1,y1,x2,y2,val_split=0.1):
        """
        Merge two datasets in chunks. So that also the validation split would match in each chunk.

        Args:
            x1 (np.array,list,dict): X-values, e.g. Coordinates in Angstroem of shape (batch,Atoms,3)
                                    Or list of np.arrays or dataforms that can be pickled.
                                    If different models require also different x-values, provide a dict matching y.
            y1 (dict):   dictionary of y values for each model.
            x2 (np.array,list,dict): X-values, e.g. Coordinates in Angstroem of shape (batch,Atoms,3)
                                    Or list of np.arrays or dataforms that can be pickled.
                                    If different models require also different x-values, provide a dict matching y.
            y2 (dict):   dictionary of y values for each model.
            val_split (list, optional): Size of the validation split. The default is 0.1.

        Returns:
            x_dict (dict): Shuffled list of the x data.
            y_dict (dict): Shuffled list of the y data.

        """
        x_dict = {}
        y_dict = {}
        models_available = sorted(list(self._models.keys()))
        y1ks = sorted(list(y1.keys()))
        y2ks = sorted(list(y2.keys()))
        for name in y1ks:
            if(name not in models_available or name not in y2ks):
                print("Error: model not available:",name)
            else:
                model_type = self._models_hyper[name][0]['general']['model_type']
                my1 = y1[name]
                my2 = y2[name]
                if(isinstance(x1,dict) == True):
                    mx1 = x1[name]
                else:
                    mx1 = x1
                if(isinstance(x2,dict) == True):
                    mx2 = x2[name]
                else:
                    mx2 = x2
                temp_x,temp_y = model_merge_data_in_chunks(model_type,mx1,my1,mx2,my2,val_split=0.1)
                x_dict.update({name:temp_x})
                y_dict.update({name:temp_y})
        
        return x_dict, y_dict
    
    
    
    # def _resample_update_active(self,x,y,indall,ind_act,chunks):
    #     #Select indall/indact = ind_unkwon
    #     ind_unknown = indall[np.isin(indall,ind_act,invert=True)]
    #     x_unknown = x[ind_unknown]
    #     y_unknown = index_data_in_y_dict(y,ind_unknown)
        
    #     #Predict unkown
    #     y_pred = self.predict(x_unknown)[0]
        
    #     #Get most dataindex of largest error
    #     maxerrind,errors = find_samples_with_max_error(y_unknown ,y_pred)
    #     #Select a chunk of the largest error index
    #     ind_add = ind_unknown[maxerrind[:chunks]]
    #     #Add new 
    #     ind_new = np.concatenate([ind_act,ind_add],axis=0)
        
    #     return ind_new,errors
    
    
        
    # def _resample_plot_stats(self,name,out_index,out_error,out_fiterr,out_testerr):
    #     # Take type and scaling into from first NN for each model
    #     model_type = self._models_hyper[name][0]['general']['model_type']
    #     _plot_resampling(model_type,os.path.join(self._directory,name,"fit_stats"),
    #                      out_index,out_error,out_fiterr,out_testerr,
    #                      self._models_hyper[name][0]['plots'])



    # def resample(self,x,y,gpu_dist,proc_async=True,random_shuffle=False,stepsize = 0.05,test_size = 0.05):
    #     """
    #     Use uncertainty sampling as active learning to effectively reduce dataset size.

    #     Args:
    #         x (np.array): Coordinates in Angstroem of shape (batch,Atoms,3)
    #         y (dict):   Dictionary of y values for each model. 
    #                     Energy in Bohr, Gradients in Hatree/Bohr. NAC in 1/Hatree.
    #                     Units are cast for fitting into eV/Angstroem.
    #         gpu_dist (dict):    Dictionary with same modelname and list of GPUs for each NN. Default is {}
    #                             Example {'nac' : [0,0] } both NNs for NAC on GPU:0
    #         proc_async (bool, optional): Try to run parallel. Default is True.    
    #         random_shuffle (bool, optional): Whether to shuffle data before fitting. Default is False. 
    #         stepsize (float, optional): Fraction of the original dataset size to add during each iteration. Defaults to 0.05.
    #         test_size (float, optional): Fraction of test set which is kept out of sampling. Defaults to 0.05.

    #     Returns:
    #         out_index (list): List of np.array of used indices from original data for each iteration.
    #         out_error (dict): Error of the unseen data.
    #         out_fiterr (dict): Validation error of fit.
    #         out_testerr (dict): Error on test set.  

    #     """
    #     #Output stats and info
    #     out_index = []  # the used data-indices for each iteration
    #     out_error = []  # Error of total datast
    #     out_fiterr = [] # Error of validation set
    #     out_testerr = [] # Error of test set
                
    #     #Temporary set number of NN to 1
    #     numNN = self._addNN
    #     self._addNN = 1
        
    #     # Length of sets
    #     total_len = len(x)
    #     chunks = int(total_len*stepsize)
    #     testchunk = int(total_len*test_size)
    #     # Index list is used for active learning
    #     index_all = np.arange(0,total_len)
        
    #     if(random_shuffle==True):
    #         index_all = index_make_random_shuffle(index_all)
          
    #     #select active and test indices
    #     ind_test = index_all[:testchunk] 
    #     index_all = index_all[testchunk:] #Remove testset from all index
    #     ind_activ = index_all[:chunks]
        
    #     #Fix test data
    #     x_test = x[ind_test]
    #     y_test = index_data_in_y_dict(y,ind_test)
        
    #     #Start selection
    #     for i_runs in range(int(1/stepsize)):
    #         out_index.append(ind_activ)
    #         #Select active data
    #         x_active = x[ind_activ]
    #         y_active = index_data_in_y_dict(y,ind_activ)
    #         #Make fit
    #         fiterrfun = self.fit(x_active,y_active,gpu_dist,proc_async,fitmode='training',random_shuffle=False)
    #         out_fiterr.append(fiterrfun)
    #         #Get test error
    #         _, ertestd = find_samples_with_max_error(y_test,self.predict(x_test)[0])
    #         out_testerr.append(ertestd)
    #         #Resample
    #         ind_new,errs = self._resample_update_active(x,y,index_all,ind_activ,chunks)
    #         out_error.append(errs)
            
    #         #Set new acitve
    #         ind_activ = ind_new
            
    #         if(len(ind_activ)>= len(index_all)):
    #             break
    #         else:
    #             print("Next fit with length: ",len(ind_activ))
        
    #     self._addNN = numNN
        
    #     # Possible plot of results here.
    #     for name in y.keys():
    #         self._resample_plot_stats(name,
    #                                   out_index,
    #                                   [x[name] for x in out_error],
    #                                   [x[name] for x in out_fiterr],
    #                                   [x[name] for x in out_testerr])
                
        
    #     return out_index,out_error,out_fiterr,out_testerr
