# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
from sklearn.impute import IterativeImputer
import torch

from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, h5_file, is_train):
        ''' Loading data from .h5 file based on (h5_file, is_train).

        :param h5_file: (String) the path of .h5 file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        # loads data
        self.X, self.e, self.y = \
            self._read_h5_file(h5_file, is_train)

        # normalizes data
        self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _normalize(self):
        ''' Performs normalizing X data. '''
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]
    
    
class SurvivalDatasetFMR(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, xlsx_file, is_train):
        ''' Loading data from .xlsx file based on (xlsx_file, is_train).

        :param xlsx_file: (String) the path of .xlsx file
        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        self.target_names = ['A4C LS', 'LASr 2', 'EF_volume', 'LVEDD', 'LVESD', 'LVEDDi', 'LVESDi imputed', 'Biplane_LVESV_or monoplane', 
                'EDV_bp or mono', 'EDVi_bp or mono', 'LAVi', 'MVannulus_4C mm', 'Mitral annulus index, mm/m2', 
                'MV_Ee_ratio_PWD', 'RV FAC'] # 15 echo parameters
        
        self.non_echo_inputs = ['NYHA 1/2/3/4 _Dr 連', 'Age@echo', 'Female or Male', 'HEIGHT ♠♠♠♠', 'WEIGHT ♠♠♠♠', 'BSA', 'HR', 'SBP', 'DBP', 'CCI', 'Atrial fibrillation  Eric 2',	'Antiplatelet  Eric 2', 'Anticoagulant Eric 2', 'ACEi  Eric 2', 'ARB  Eric 2', 'ARNi  Eric', 'Beta-blocker Eric 2', 'CCB Eric 2', 'MRA Eric 2', 'Statins Eric 2', 'Diuretics Eric 2', 'Digoxin Eric 2', 'Nitrate Eric 2', 'HYdralazine Eric 2', 'Ivabradine Missing', 'Anti-arrhYthmic meds (class I) Eric 2', 'Anti-arrhYthmic meds (class III) Eric 2']
        
        # loads data
        self.X, self.e, self.y = \
            self._read_xlsx_file(xlsx_file, is_train)

        # normalizes data
        self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))
    

    def _read_xlsx_file(self, xlsx_file, is_train):
        ''' The function to parsing data from .xlsx file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''

        df = pd.read_excel(xlsx_file)

        def clean_value(x):
            if type(x) is float:
                return x
            else:
                if '-' in x:  # If range like '1-2'
                    nums = list(map(float, x.split('-')))
                    return np.mean(nums)
                elif ',' in x:  # If range like '2,3'
                    nums = list(map(float, x.split(',')))
                    return np.mean(nums)
                else:  # Single number
                    return float(x)

        df['NYHA 1/2/3/4 _Dr 連'] = df['NYHA 1/2/3/4 _Dr 連'].apply(clean_value)
        df['Female or Male'] = df['Female or Male'].map({'M': 0, 'F': 1})
        
        imputer = IterativeImputer(random_state=2024)
        df[self.non_echo_inputs] = imputer.fit_transform(df[self.non_echo_inputs])
        
        # imputer = IterativeImputer(random_state=2024)
        # df[self.target_names] = imputer.fit_transform(df[self.target_names])
        
        # imputer = IterativeImputer(random_state=2024)
        # df[[i+'_pred' for i in self.target_names]] = imputer.fit_transform(df[[i+'_pred' for i in self.target_names]])
        
        df = df.dropna(subset=self.non_echo_inputs)
        df = df.dropna(subset=self.target_names)
        df = df.dropna(subset=[i+'_pred' for i in self.target_names])
        
        X = df[self.target_names+self.non_echo_inputs].to_numpy().astype(np.float32) # use gt echo
        # X = df[[i+'_pred' for i in self.target_names]+self.non_echo_inputs].to_numpy().astype(np.float32) # use predicted echo
        
        e = df['4) LVAD/HTx/CVD=1'].to_numpy().reshape(-1, 1)
        y = df['4) Total FU for CVD/death equivalent, yr'].to_numpy().reshape(-1, 1)
        
        np.random.seed(2024)
        indices = np.random.permutation(X.shape[0])
        split_idx = int(0.8 * X.shape[0])
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        e_train, e_test = e[train_idx], e[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
            
        if is_train:
            return X_train, e_train, y_train
        else:
            return X_test, e_test, y_test

    # def _normalize(self):
    #     ''' Performs normalizing X data. '''
    #     self.X = (self.X-self.X.min(axis=0)) / \
    #         (self.X.max(axis=0)-self.X.min(axis=0))

    def _normalize(self):
        ''' Performs Z-normalization (standardization) on X data. '''
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
    
    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]