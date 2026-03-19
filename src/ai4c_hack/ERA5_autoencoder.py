#!/usr/bin/env python
# ## AI4Climate ML tutorial - Building a machine learning model with gridded data
# * Author: Stephen Haddad
# * Affiliation: UK Met Office
# * History: 1.0
# * Last update: 2026-03-16
# * © British Crown Copyright 2017-2026, Met Office. Please see LICENSE.md for license details.
import pathlib
import os
import datetime
import json
import re
import argparse

import numpy 
import xarray

import sklearn
import sklearn.preprocessing

import mlflow

import torch



class WeatherbenchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, time_period, is_train=True):
        self._is_train=is_train

        self._ds_norm = xarray.open_zarr(data_dir)
        self._ds_norm = self._ds_norm.loc[{'time':slice(*time_period)}]


        self._time_list = (self._ds_norm[ list(self._ds_norm.keys())[0] ]['time'].values)
        self.num_channels = len(self._ds_norm.data_vars)*len(self._ds_norm['level'])


    def __str__(self):
        return str(self._wb_ds)

    def __repr_html__(self):
        return self._wb_ds.__rept_html__()

    def __len__(self):
        return len(self._time_list)

    def __getitem__(self, idx):
        selected_time = self._ds_norm.time[idx].values
        select_ds = self._ds_norm.loc[{'time': selected_time}]
        if type(idx) == int:
            reshape_args = (self.num_channels, len(select_ds['lat']),len(select_ds['lon']) )
        else:
            reshape_args = (-1, self.num_channels, len(select_ds['lat']),len(select_ds['lon']) )
        select_array = numpy.stack(
            [select_ds[v1].to_numpy() for v1 in select_ds.data_vars],
            axis=1).reshape(reshape_args)


        select_tensor = torch.tensor(
            select_array,
            dtype=torch.float32,
        )
        return select_tensor


class Era5AutoEncoder(torch.nn.Module):
    def __init__(self, num_channels):
        super(Era5AutoEncoder, self).__init__()

        # we have "hard coded" a lot of the architecture hyperparameters in our model class. 
        # Usually you want want to make these arguments for the class so you can vary hyperparameters more easily.
        # Hard coding here makes it easier to follow the architecture definition in the tutorial
        self.num_channels = num_channels
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.num_channels, 
                            out_channels=16, 
                            kernel_size=3, 
                            padding=1,
                           ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(in_channels=16, 
                            out_channels=32, 
                            kernel_size=3, 
                            padding=1,
                           ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Flatten(1,-1)
        )
        self._latent_array_dims = (-1,32,8,16)
        self._decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2,stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.num_channels, kernel_size=2,stride=2),
            torch.nn.ReLU(),
            torch.nn.Sigmoid(),   
        )



    def forward(self, x):

        # Get latent representation
        latent = self._encoder(x)

        # Reconstruct input
        reconstructed = self._decoder(latent.view(self._latent_array_dims))

        return reconstructed        

def run_training(num_epochs, train_loader, val_loader, num_channels):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device
    
    # Initialize model and move to device
    ae_model = Era5AutoEncoder(num_channels).to(device)
    
    # Loss function and optimizer
    loss_function = torch.nn.L1Loss()
    # criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), 
                                 lr=1e-4)
    for epoch_num in range(num_epochs):
        print(epoch_num)
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        
        for batch_ix, X_batch in enumerate(wb_train_loader):
            if (batch_ix % 20) == 0:
                print(batch_ix)
            optimizer.zero_grad()
            predictions = ae_model.forward(X_batch.to(device))
            loss_batch = loss_function(predictions, X_batch.to(device))
            loss_batch.backward()
            optimizer.step()
            epoch_train_loss += loss_batch.to('cpu').item()
        epoch_train_loss /= len(wb_train_loader)
        print(epoch_train_loss)
    return device, ae_model
    
    
def get_config(config_path):
    with open (config_path,'r') as tutorial_config:
        tutorial_config = json.load(tutorial_config)
    return tutorial_config

def get_platform_dir(select_platform, config):
    try:
        root_path = pathlib.Path(config['default_dirs'][select_platform]) / 'weatherbench'
    except KeyError:
        root_path = pathlib.Path(os.environ['HOME']) / 'weatherbench'
    return root_path

def get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Training script arguments"
    )

    parser.add_argument(
        "--config-path",
        dest="config_path",
        type=pathlib.Path,
        required=True,
        help="Path to the config file"
    )

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the dataset (zarr format)"
    )

    parser.add_argument(
        "--model-out-path",
        dest="model_out_path",
        type=pathlib.Path,
        required=True,
        help="Path to the dataset (zarr format)"
    )
    
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        required=True,
        help="Number of data points per mini-batch"
    )

    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        type=int,
        required=True,
        help="Number epochs to run training loop for"
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        required=True,
        help="Learning rate for training"
    )

    args = parser.parse_args()

    # return as dictionary (W behavior)
    return args

def main():
    cmd_args = get_cmd_args()
    tutorial_config = get_config(args.config_path)
    current_platform = tutorial_config['platform']
    
    root_data_dir = get_platform_dir(current_platform, tutorial_config)
    resolution_dict = {5.625: '5.625deg'}
    weatherbench_dir = root_data_dir / resolution_dict[5.625]
    wb_arco_path = root_data_dir / 'wb_arco'

    batch_size = cmd_args.batch_size

    wb_train_ds = WeatherbenchDataset(wb_arco_path, 
                           (datetime.datetime(1980,1,1,0,0), datetime.datetime(1981,11,1,0,0)),                        
                           is_train=True,
                          )
    
    wb_val_ds = WeatherbenchDataset(wb_arco_path, 
                                      (datetime.datetime(1981,11,2,0,0), datetime.datetime(1982,1,1,0,0)), 
                                      is_train=False,
                                     )

    wb_train_loader = torch.utils.data.DataLoader(wb_train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                          )
    wb_val_loader = torch.utils.data.DataLoader(wb_val_ds,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                        )

    num_epochs = cmd_args.num_epochs

    device, ae_model = run_training(num_epochs, 
                                    wb_train_loader, 
                                    wb_val_loader, 
                                    wb_train_ds.num_channels)

    print(f'saving model to {cmd_args.model_out_path}')
    torch.save(ae_model, cmd_args.model_out_path)

if __name__ == '__main__':
    main()