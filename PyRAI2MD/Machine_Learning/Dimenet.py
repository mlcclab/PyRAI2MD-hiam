#####################################################
#
# DimeNet model for NAC prediction
#
# Author Sijin Ren
# Apr 19 2023
#
######################################################

import os
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from pymatgen.core.periodic_table import Element
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus, DimeNet
from torch_geometric.data import Data
from torch.optim import lr_scheduler


class DimenetNAC:

    def __init__(self, param):
        self.param = deep_update(DEFAULT_PARAM, param)  # Jingbai: update the default param with the input param
        self.batch_size = self.param['batch_size']
        self.val_size = self.param['val_size']
        self.criterion = self.param['criterion']
        self.lr_start = self.param['lr_start']
        self.nepochs = self.param['nepochs']
        self.model_param = self.param['model_param']
        self.model_path = self.param['model_path']
        self.shuffle = self.param['shuffle']
        self.gpu = self.param['gpu']
        self.model = None
        self.data = None
        self.nac_size = None
        self.train_loader = None
        self.val_loader = None
        self.scheduler = None
        self.optimizer = None
        self.criterion = None
        self.device = None

        self.set_device()
        self.set_model()
        self.set_criterion()
        self.set_optimizer()
        self.set_scheduler()

    def set_device(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and self.gpu else "cpu")

    def set_model(self):

        # model_param = Param['model_param']  # Jingbai: we can put Param in __init__ then use self.param
        if self.param['model_type'] == 'Dimenet':
            self.model = DimeNet(
                hidden_channels=self.param['model_param']['hidden_channels'],
                out_channels=self.param['model_param']['out_channels'],
                num_blocks=self.param['model_param']['num_blocks'],
                num_bilinear=self.param['model_param']['num_bilinear'],
                num_spherical=self.param['model_param']['num_spherical'],
                num_radial=self.param['model_param']['num_radial']
            )
        elif self.param['model_type'] == 'Dimenet++':
            self.model = DimeNetPlusPlus(
                hidden_channels=self.param['model_param']['hidden_channels'],
                out_channels=self.param['model_param']['out_channels'],
                num_blocks=self.param['model_param']['num_blocks'],
                int_emb_size=self.param['model_param']['int_emb_size'],
                basis_emb_size=self.param['model_param']['basis_emb_size'],
                out_emb_channels=self.param['model_param']['out_emb_channels'],
                num_spherical=self.param['model_param']['num_spherical'],
                num_radial=self.param['model_param']['num_radial'],
            )

    def set_criterion(self):
        if self.param['criterion'] == 'MAE':
            self.criterion = torch.nn.L1Loss()
        elif self.param['criterion'] == 'MSE':
            self.criterion = torch.nn.MSELoss()

    def set_scheduler(self):
        # scheduler for learning rate
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def set_optimizer(self):
        if self.param['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_start)

    def set_data(self, data):
        # Jingbai: PyRAI2MD will read the .json file and sent a dict to prepare pytorch-geometric data
        # data = json.load(open(self.data_file))
        df = pd.DataFrame.from_dict(data)
        dataset = []
        for i in df.index:
            nac_np = np.array(df.loc[i, 'nac']).squeeze()
            xyz_np = np.array(df.loc[i, 'xyz'])
            symbol = xyz_np[:, 0]
            z = [Element(s).Z for s in symbol]
            coord = xyz_np[:, 1: 4].astype(np.float64)
            node_pos = torch.tensor(coord, dtype=torch.float, device=self.device)  # Jingbai: create tensor on device
            node_z = torch.tensor(z, dtype=torch.int, device=self.device)  # avoid .to(device) in train loop
            y = torch.tensor(nac_np, dtype=torch.float, device=self.device)
            data = Data(z=node_z, y=y, pos=node_pos)
            dataset.append(data)

        self.data = dataset
        self.nac_size = self.data[0].y.size()[0] * self.data[0].y.size()[1]

    def set_loaders(self):
        # train/val split, and prepare loader
        dataset = self.data
        val_size = self.val_size
        dataset = shuffle(dataset, random_state=2)
        val_dataset = dataset[:val_size]
        train_dataset = dataset[val_size:]
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def phaseless_loss(self, pred, target):
        target1 = target.reshape(-1, self.nac_size)
        target2 = torch.neg(target).reshape(-1, self.nac_size)
        pred = pred.reshape(-1, self.nac_size)
        losses = []
        for i in range(len(target1)):
            loss1 = self.criterion(pred[i], target1[i])
            loss2 = self.criterion(pred[i], target2[i])
            losses.append(min(loss1, loss2))
        loss = sum(losses) / len(losses)
        return loss

    def train_epoch(self, loader):
        self.model.train()
        for data in loader:
            self.optimizer.zero_grad()
            out = self.model(data.z, data.pos, data.batch)
            pred = torch.flatten(out)
            target = torch.flatten(data.y)
            loss = self.phaseless_loss(pred, target)
            loss.backward()
            self.optimizer.step()

    def val_epoch(self, loader):
        self.model.eval()
        pred_all_batches = []
        target_all_batches = []
        for data in loader:
            out = self.model(data.z, data.pos, data.batch)
            pred = torch.flatten(out.cpu()).detach().numpy().tolist()
            pred_all_batches = pred_all_batches + pred
            target = torch.flatten(data.y.cpu()).detach().numpy().tolist()
            target_all_batches = target_all_batches + target
        pred_all = torch.tensor(pred_all_batches)
        target_all = torch.tensor(target_all_batches)
        pred_loss = self.phaseless_loss(pred_all, target_all)
        return target_all_batches, pred_all_batches, pred_loss

    def fit(self, data):
        # Jingbai: I moved the following steps here, so we set up training data after calling fit().
        self.set_data(data)
        self.set_loaders()

        n_epochs = self.nepochs
        train_error_epoches = []
        val_error_epoches = []
        best_df = pd.DataFrame()  # Jingbai: this variable is not used
        best_val_error = 100000
        count = 0
        self.model.to(self.device)
        for epoch in range(1, n_epochs + 1):
            count = count + 1
            print('_____________________________epoch', epoch)
            self.train_epoch(self.train_loader)
            train_target, train_pred, train_error = self.val_epoch(self.train_loader)
            val_target, val_pred, val_error = self.val_epoch(self.val_loader)
            train_error_epoches.append(train_error)
            val_error_epoches.append(val_error)
            if val_error < best_val_error or count == 1:
                print('updated')
                self.save_model(epoch)
                best_val_error = val_error
                val_target = np.array(val_target).reshape(-1, self.nac_size)  # Jingbai: this variable is not used
                val_pred = np.array(val_pred).reshape(-1, self.nac_size)  # Jingbai: this variable is not used
            print(f'{train_error:.10f},{val_error:.10f}')
        # return ferr dictionary with 'nac' key
        ferr = {'nac': [float(best_val_error)]}
        return ferr

    def set_loader_test(self, xyz_list):
        # xyz_list:[nmolecule,natom,4]
        dataset_pred = []
        for j in xyz_list:  # loop through molecules
            xyz_np = np.array(j)
            symbol = xyz_np[:, 0]
            z = [Element(s).Z for s in symbol]
            coord = xyz_np[:, 1: 4].astype(np.float64)
            node_pos = torch.tensor(coord, dtype=torch.float, device=self.device)  # Jingbai: create tensor on device
            node_z = torch.tensor(z, dtype=torch.int, device=self.device)  # avoid .to(device) in train loop
            data = Data(z=node_z, pos=node_pos)
            dataset_pred.append(data)
        test_loader = DataLoader(dataset_pred, batch_size=1, shuffle=False)

        return test_loader

    def predict(self, xyz_list):
        self.model.eval()
        pred_all_batches = []
        loader = self.set_loader_test(xyz_list)
        for data in loader:
            out = self.model(data.z, data.pos, data.batch).reshape(-1, 3)
            pred = torch.flatten(out.cpu()).detach().numpy().tolist()
            pred_all_batches = pred_all_batches + pred
        pred_all = torch.tensor(pred_all_batches)
        pred_all = pred_all.reshape((-1, int(self.nac_size / 3), 3)).tolist()

        # Jingbai: the mean_dict['nac'] stores the mean value of two models, but we only have one at the moment
        mean_dict = {
            'nac': pred_all,  # pred_all_1 is a list containing all predicted NACs in [nmolecules,natoms,4]
        }

        # Jingbai: the std_dict['nac'] stores the std value of two models, but we only have one, so std is 0.
        std_dict = {
            'nac': [0],
        }

        return mean_dict, std_dict

    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save({
            'epoch': epoch,
            'nac_size': self.nac_size,  # Jingbai this var is needed, otherwise loading a model won't know the nac_size
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, '%s/nac' % self.model_path)

    def load_model(self):
        checkpoint = torch.load('%s/nac' % self.model_path, map_location=self.device)
        self.nac_size = checkpoint['nac_size']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)


# net = Dimenet_nac(Param)
# a=net.predict(net.val_loader)

## Jingbai: here store some default parameters
DEFAULT_PARAM = {
    'model_type': 'Dimenet',
    # 'data_file': 'nac_data.json',
    'optimizer': 'Adam',
    'batch_size': 30,
    'val_size': 2000,
    'criterion': 'MAE',
    'model_param': {
        'hidden_channels': 256,
        'out_channels': 36,
        'num_blocks': 6,
        'num_bilinear': 8,
        'num_spherical': 7,
        'num_radial': 6
    },
    'lr_start': 0.001,
    'nepochs': 200,
    'model_path': './best_model',
    'shuffle': False,  # Jingbai: shuffle data loader during training
    'gpu': 0  # Jingbai: user can choose to not use gpu if set this to 0
}


def deep_update(a, b):
    # recursively update a with b
    for key, val in b.items():
        if key in a.keys():
            if isinstance(val, dict) and isinstance(a[key], dict):
                a[key] = deep_update(a[key], val)
            else:
                a[key] = val
        else:
            a[key] = val

    return a
