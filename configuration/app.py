import os
import torch
import logging

DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE            = True
EXPLORE         = True
NOISE           = {'mean': 0., 'std': 1.0}
WN              = {'mean': 0., 'std': 0.10}
AVAIL_GPUS      = min(1, torch.cuda.device_count())
NUM_WORKERS     = int(os.cpu_count() / 2)
LR              = 0.0005
B1              = 0.5
B2              = 0.999
NCRITIC_DISC    = 1
NCRITIC_GEN     = 5
ALPHA           = 0.10
BETA            = 10.
LAMBDA          = 10.
LAMBDA_1        = 10.E+3
LAMBDA_2        = 10.E+3
LOGGING_LEVEL   = logging.INFO
LAMBDA_ALI      = 1.
LOGGING_FORMAT  = '%(levelname)s:%(message)s'
LAMBDA_IDENTITY = 10.
LAMBDA_GP       = 10.
LAMBDA_KL       = 0.10
GAMMA           = 1000.
OPTIMAL_VAL     = 0.6108643020548934
LAMBDA_CONSISTENCY = 10.
            
logger          = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

formatter       = logging.Formatter(LOGGING_FORMAT)

stream_handler  = logging.StreamHandler()
stream_handler.setFormatter(formatter)


logger.addHandler(stream_handler)
