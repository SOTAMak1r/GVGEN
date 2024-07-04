# optimizer
from torch.optim import SGD, Adam  # RAdam, 1.8 doesn't have Radam
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ExponentialLR


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_optimizer(hparams, models, rate=1):
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    if hparams.optimizer == 'sgd':
        optimizer = SGD(
            parameters, 
            lr=hparams.lr*rate,
            momentum=hparams.momentum, 
            weight_decay=hparams.weight_decay
        )
    elif hparams.optimizer == 'adam':
        optimizer = Adam(
            parameters, 
            lr=hparams.lr*rate, 
            eps=eps,
            weight_decay=hparams.weight_decay
        )
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 2e-7
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(
            optimizer, 
            milestones=hparams.decay_step,
            gamma=hparams.decay_gamma
        )
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=hparams.num_epochs, 
            eta_min=eps,
        )
    else:
        scheduler = ExponentialLR(
            optimizer, 
            gamma=hparams.decay_gamma
        )

        scheduler = {
            "interval": "epoch",
            "scheduler": scheduler,
        }
    return scheduler