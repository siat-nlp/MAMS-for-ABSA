from torch import optim
import adabound

def make_optimizer(config, model):
    mode = config['mode']
    config = config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    opt = {
        'sgd': optim.SGD,
        'adadelta': optim.Adadelta,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'adagrad': optim.Adagrad,
        'asgd': optim.ASGD,
        'rmsprop': optim.RMSprop,
        'adabound': adabound.AdaBound
    }
    if 'momentum' in config:
        optimizer = opt[config['optimizer']](model.parameters(), lr=lr, weight_decay=weight_decay, momentum=config['momentum'])
    else:
        optimizer = opt[config['optimizer']](model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer