# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
import shap
import matplotlib.pyplot as plt
import pandas as pd

from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset, SurvivalDatasetFMR
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger

def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device)
    criterion = NegativeLogLikelihood(config['network'], device).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
    # train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    # test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)

    train_dataset = SurvivalDatasetFMR(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDatasetFMR(config['train']['h5_file'], is_train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            risk_pred = model(X)
            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            with torch.no_grad():
                risk_pred = model(X)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:
                    flag += 1
                    if flag >= patience:
                        return best_c_index
                
            if flag == 0:
                ## test the shap function
                inputs, _, _ = next(iter(test_loader))  # Assuming your loader returns (data, labels)

                # Optional: If you want to work with a subset
                X_batch = inputs[:100]  # shape: (batch_size, num_features)
                X_batch = X_batch.to(device)

                # DeepExplainer needs a small background sample from training data
                background, _, _ = next(iter(train_loader))
                background = background[:292]
                background = background.to(device)

                # Explain the model
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X_batch).squeeze()


                # Convert tensor to DataFrame for better plotting
                feature_names = train_dataset.target_names+train_dataset.non_echo_inputs
                # feature_names = [f"feature_{i}" for i in range(X_batch.shape[1])]
                X_df = pd.DataFrame(X_batch.cpu().numpy(), columns=feature_names)

                # Plot summary
                shap.summary_plot(shap_values, X_df, show=False)
                plt.savefig("shap_summary.png", dpi=500, bbox_inches='tight')
                plt.close()
                ##
       
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index

if __name__ == '__main__':
    # global settings
    logs_dir = 'logs'
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = 'configs'
    # params = [
    #     ('Simulated Linear', 'linear.ini'),
    #     ('Simulated Nonlinear', 'gaussian.ini'),
    #     ('WHAS', 'whas.ini'),
    #     ('SUPPORT', 'support.ini'),
    #     ('METABRIC', 'metabric.ini'),
    #     ('Simulated Treatment', 'treatment.ini'),
    #     ('Rotterdam & GBSG', 'gbsg.ini')
    #     ]
    params = [
        ('FMR Cohort', 'fmr.ini')
    ]
    patience = 500
    # training
    headers = []
    values = []
    for name, ini_file in params:
        logger.info('Running {}({})...'.format(name, ini_file))
        best_c_index = train(os.path.join(configs_dir, ini_file))
        headers.append(name)
        values.append('{:.6f}'.format(best_c_index))
        print('')
        logger.info("The best valid c-index: {}".format(best_c_index))
        logger.info('')
    # prints results
    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)

