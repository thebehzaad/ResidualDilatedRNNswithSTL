from math import sqrt

import torch

def get_config(interval):
    config = {
        'device': ("cuda" if torch.cuda.is_available() else "cpu"),        
        'add_nl_layer': True,
        'rnn_cell_type': 'LSTM',
        'learning_rate': 1e-3,
        'learning_rates': ((10, 1e-4)),
        'num_of_train_epochs': 20,
        'num_of_categories': 6,  # in data provided
        'batch_size': 1024,
        'gradient_clipping': 20,
        'lr_anneal_rate': 0.5,
        'lr_anneal_step': 5
    }

    if interval == 'Quarterly':
        config.update({
            'chop_val': 72,
            'variable': "Quarterly",
            'dilations': ((1,4),),
            'state_hsize': 40,
            'seasonality': 4,
            'input_size': 1,
            'output_size': 8
        })
    elif interval == 'Monthly':
        config.update({
            'chop_val': 72,
            'variable': "Monthly",
            'dilations': ((1, 3), (6, 12)),
            'state_hsize': 50,
            'seasonality': 12,
            'input_size': 12,
            'output_size': 18
        })
    elif interval == 'Daily':
        config.update({
            'chop_val': 200,
            'variable': "Daily",
            'dilations': ((1, 7), (14, 28)),
            'state_hsize': 50,
            'seasonality': 7,
            'input_size': 7,
            'output_size': 14
        })
    elif interval == 'Yearly':
        config.update({
            'chop_val': 25,
            'variable': "Yearly",
            'dilations': ((1, 2), (2, 6)),
            'state_hsize': 30,
            'seasonality': 1,
            'input_size': 4,
            'output_size': 6
        })
    else:
        print("I don't have that config. :(")

    return config
