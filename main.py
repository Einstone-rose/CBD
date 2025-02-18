import numpy as np
from datetime import datetime
import argparse
from utils.util import _logger
from models.base_model_mae_full import BaseModel  # base_model_mae_v2
from data_provider.dataloader import data_generator
from train_step.single_pretrain_mae import Trainer as SinglePretrainTrainer  # mae
from train_step.single_finetune import Trainer as SingleFinetuneTrainer
from train_step.single_linear import Trainer as SingleLinearTrainer
import os
import torch
import random


start_time = datetime.now()
parser = argparse.ArgumentParser()
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp17', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run_on_ps', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=2024, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str,  # self_supervised  finetune_1p  train_linear_1p
                    help='Modes of choice: random_init, supervised, self_supervised, finetune, train_linear')
parser.add_argument('--selected_dataset', default='PhonemeSpectra', type=str,
                    help='Dataset of choice: ArticularyWordRecognition, HAR, UWaveGestureLibraryAll, ECG5000, MotorImagery, FingerMovements')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args, unknown = parser.parse_known_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'SPANet'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
#####################################################

# option = f'_lambda{configs.lam}_mask{configs.mask_ratio}_layer{configs.encoder_layer_num}_dlayer{configs.decode_layer_num}_fd{configs.freq_decode_layer}_wo_fd'
option = f'lr_{configs.lr}_lambda{configs.lam}_layer{configs.encoder_layer_num}_dlayer{configs.decode_layer_num}_fd{configs.freq_decode_layer}'
experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_{option}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug(f'Seed:    {SEED}')
logger.debug(f'mask:    {configs.mask_ratio}')
logger.debug(f'lam:     {configs.lam}')
logger.debug(f'lr:      {configs.lr}')
logger.debug(f'ft_lt:   {configs.ft_lr}')
logger.debug(f'pre_epo: {configs.num_epoch}')
logger.debug(f'ft_epo:  {configs.finetune_epoch}')
logger.debug(f'head:    {configs.encode_head_num}')
logger.debug(f'layer:   {configs.encoder_layer_num}')
logger.debug(f'd_head:  {configs.decode_head_num}')
logger.debug(f'd_layer: {configs.decode_layer_num}')
logger.debug(f'fd_layer:{configs.freq_decode_layer}')
logger.debug("=" * 45)


start_epoch = 1
# Load datasets
data_path = f'./dataset/{data_type}'
train_dl, val_dl, test_dl = data_generator(data_path, configs, data_type, training_mode)
logger.debug("Data loaded ...")
acc_list = []
loss_list = []

if training_mode == "self_supervised":
    model = BaseModel(configs, 'pretrain').to(device)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    loss_list = SinglePretrainTrainer(model, model_optimizer, train_dl, device, logger, args, configs, experiment_log_dir, SEED, start_epoch, training_mode)
elif "finetune" in training_mode:
    start_epoch = 5
    load_path = f'./experiments_logs/{experiment_description}/{run_description}/self_supervised_seed_{SEED}_{option}/saved_models'
    acc_list = SingleFinetuneTrainer(train_dl, val_dl, test_dl, device, logger, args, configs, experiment_log_dir, start_epoch, load_path)
elif "train_linear" in training_mode:
    print(f'mask len is {int(configs.features_len * configs.mask_ratio)}')
    start_epoch = 5
    load_path = f'./experiments_logs/{experiment_description}/{run_description}/self_supervised_seed_{SEED}_{option}/saved_models'
    trainer, acc_list = SingleLinearTrainer(train_dl, val_dl, test_dl, device, logger, args, configs, experiment_log_dir, start_epoch, load_path)
elif training_mode == "supervised":
    trainer, acc_list = SupervisedTrainer(train_dl, val_dl, test_dl, device, logger, args, configs, experiment_log_dir)

logger.debug(f"Training time is : {datetime.now()-start_time}")
if acc_list is not None and len(acc_list) > 0:
    logger.debug(f'acc list: {acc_list}')



