import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from DRNN import DRNN


class ResidualDRNN_Main(nn.Module):
    def __init__(self, config):
        super(ResidualDRNN_Main, self).__init__()
        
        self.config = config
        self.add_nl_layer = self.config['add_nl_layer']

        self.nl_layer = nn.Linear(config['state_hsize'],config['state_hsize'])
        self.act = nn.Tanh()
        
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])
        self.logistic = nn.Sigmoid()

        self.resid_drnn = ResidualDRNN(self.config)

    def forward(self, train, resid, trend, seasonal, mean, std, val, info_cat):
        
        train = train.float()
        resid = resid.float()
        trend = trend.float()
        seasonal = seasonal.float()
        mean = mean.float()
        std = std.float()
        #--------------------------------------------------------------------------------------        
        window_input_list = []
        window_output_list = []
        for i in range(self.config['input_size'] - 1, train.shape[1]):
            input_window_start = i + 1 - self.config['input_size']
            input_window_end = i + 1

            train_window_input = resid[:, input_window_start:input_window_end]
            
            train_cat_window_input = torch.cat((train_window_input, info_cat), dim=1)
            window_input_list.append(train_cat_window_input)

            output_window_start = i + 1
            output_window_end = i + 1 + self.config['output_size']

            if i < train.shape[1] - self.config['output_size']:
                train_window_output = resid[:, output_window_start:output_window_end]
                window_output_list.append(train_window_output)

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

        self.train()
        network_pred = self.series_forward(window_input[:-self.config['output_size']])
        network_act = window_output
        
        
        self.eval()
        network_output_non_train = self.series_forward(window_input)

        # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS----------------------------------
        hold_out_pred = network_output_non_train[-1]                
        
        # DENORMALIZATION
        hold_out_pred = hold_out_pred*std.reshape(-1,1)+mean.reshape(-1,1)
              
        # ADDING BACK THE TREND COMPONENT 
        last_step=trend[:,-1:]-trend[:,-2:-1]
        for i in range(hold_out_pred.shape[1]):
            hold_out_pred[:,i:i+1] = hold_out_pred[:,i:i+1]+trend[:,-1:]
        
        # ADDING BACK THE SEASONALITY COMPONENT
        last_season=seasonal[:,-self.config['seasonality']:]
        for i in range(hold_out_pred.shape[1]):
            j=i%self.config['seasonality']
            hold_out_pred[:,i:i+1] = hold_out_pred[:,i:i+1]+last_season[:,j:j+1]
               
        hold_out_act = val
        #---------------------------------------------------------------------------------------------------------------
        
        self.train()
        # RETURN 
        return network_pred, \
               network_act, \
               network_output_non_train, \
               hold_out_pred, \
               hold_out_act
               

    def series_forward(self, data):
        data = self.resid_drnn(data)
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data


class ResidualDRNN(nn.Module):
    def __init__(self, config):
        super(ResidualDRNN, self).__init__()
        self.config = config

        layers = []
        for grp_num in range(len(self.config['dilations'])):

            if grp_num == 0:
                input_size = self.config['input_size'] + self.config['num_of_categories']
            else:
                input_size = self.config['state_hsize']

            l = DRNN(input_size,
                     self.config['state_hsize'],
                     n_layers=len(self.config['dilations'][grp_num]),
                     dilations=self.config['dilations'][grp_num],
                     cell_type=self.config['rnn_cell_type'])

            layers.append(l)

        self.rnn_stack = nn.Sequential(*layers)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
            input_data = out
        return out
