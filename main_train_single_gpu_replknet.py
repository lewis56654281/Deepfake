import os
import time
import datetime
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.tensorboard import SummaryWriter
from core.dsproc_mcls import MultiClassificationProcessor
from core.mengine import TrainEngine
from toolkit.dtransform import create_transforms_inference, transforms_imagenet_train
from toolkit.yacs import CfgNode as CN
from timm.utils import ModelEmaV2

import warnings

warnings.filterwarnings("ignore")

# check
print(torch.__version__)
print(torch.cuda.is_available())

# init
cfg = CN(new_allowed=True)

# dataset dir
ctg_list = '/home/tiancheng/Deepfake/DeepFakeDefenders/dataset/phase1/label.txt'
train_list = '/home/tiancheng/Deepfake/DeepFakeDefenders/dataset/phase1/train.txt'
val_list = '/home/tiancheng/Deepfake/DeepFakeDefenders/dataset/phase1/val.txt'

# : network
cfg.network = CN(new_allowed=True)
cfg.network.name = 'replknet'
cfg.network.class_num = 2
cfg.network.input_size = 384

# : train params
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

cfg.train = CN(new_allowed=True)
cfg.train.resume = True  # ���改为True
cfg.train.resume_path = '/home/tiancheng/Deepfake/DeepFakeDefenders/output/2024-10-26-00-01-16_replknet_to_competition_BinClass/checkpoint_epoch_3.pth'  # 设置检查点路径
cfg.train.params_path = ''
cfg.train.batch_size = 16
cfg.train.epoch_num = 20
cfg.train.epoch_start = 4  # 设置为您想继续训练的起始epoch
cfg.train.worker_num = 8
cfg.train.use_full_data = True  # 设置为False时使用部分数据
cfg.train.partial_data_ratio = 1  # 使用10%的数据进行初步训练

# : optimizer params
cfg.optimizer = CN(new_allowed=True)
cfg.optimizer.name = 'adamw'  # 或 'adam' 或 'sgd'
cfg.optimizer.lr = 1e-4 * 1
cfg.optimizer.weight_decay = 1e-2
cfg.optimizer.momentum = 0.9  # 仅用于 SGD
cfg.optimizer.beta1 = 0.9
cfg.optimizer.beta2 = 0.999
cfg.optimizer.eps = 1e-8

# : scheduler params
cfg.scheduler = CN(new_allowed=True)
cfg.scheduler.min_lr = 1e-6

# init path
task = 'competition'
log_root = 'output/' + datetime.datetime.now().strftime("%Y-%m-%d") + '-' + time.strftime(
    "%H-%M-%S") + '_' + cfg.network.name + '_' + f"to_{task}_BinClass"

if not os.path.exists(log_root):
    os.makedirs(log_root)
writer = SummaryWriter(log_root)

# create engine
train_engine = TrainEngine(0, 0, DDP=False, SyncBatchNorm=False)
train_engine.create_env(cfg)

# 在这里检查优化器是否已创建
if train_engine.optimizer_ is None:
    raise ValueError("Optimizer has not been created. Check the create_env method.")

# create transforms
transforms_dict = {
    0: transforms_imagenet_train(img_size=(cfg.network.input_size, cfg.network.input_size)),
    1: transforms_imagenet_train(img_size=(cfg.network.input_size, cfg.network.input_size), jpeg_compression=1),
}

transforms_dict_test = {
    0: create_transforms_inference(h=512, w=512),
    1: create_transforms_inference(h=512, w=512),
}

transform = transforms_dict
transform_test = transforms_dict_test

# create dataset
trainset = MultiClassificationProcessor(transform)
trainset.load_data_from_txt(train_list, ctg_list, '/home/tiancheng/Deepfake/DeepFakeDefenders/dataset/phase1/train_dataset/')

valset = MultiClassificationProcessor(transform_test)
valset.load_data_from_txt(val_list, ctg_list, '/home/tiancheng/Deepfake/DeepFakeDefenders/dataset/phase1/val_dataset/')

# 根据配置决定是否使用部分数据
def load_partial_data(dataset, ratio):
    full_size = len(dataset)
    partial_size = int(full_size * ratio)
    return torch.utils.data.Subset(dataset, range(partial_size))

# 根据配置决定是否使用部分数据
if not cfg.train.use_full_data:
    trainset = load_partial_data(trainset, cfg.train.partial_data_ratio)
    valset = load_partial_data(valset, cfg.train.partial_data_ratio)

# create dataloader
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=cfg.train.batch_size,
                                           num_workers=cfg.train.worker_num,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=valset,
                                         batch_size=cfg.train.batch_size,
                                         num_workers=cfg.train.worker_num,
                                         shuffle=False,
                                         pin_memory=True,
                                         drop_last=False)

train_log_txtFile = log_root + "/" + "train_log.txt"
f_open = open(train_log_txtFile, "w")

# train & Val & Test
best_test_mAP = 0.0
best_test_idx = 0.0
ema_start = True
train_engine.ema_model = ModelEmaV2(train_engine.netloc_).cuda()

# 创建并加载 EMA 模型
if cfg.train.resume and cfg.train.resume_path:
    ema_checkpoint_path = cfg.train.resume_path.replace('checkpoint_', 'ema_checkpoint_')
    if os.path.exists(ema_checkpoint_path):
        ema_checkpoint = torch.load(ema_checkpoint_path)
        if 'ema_model_state_dict' in ema_checkpoint:
            train_engine.ema_model.load_state_dict(ema_checkpoint['ema_model_state_dict'])
            print(f"Loaded EMA model from {ema_checkpoint_path}")

# 在训练循环开始前，打印使用的数据量
print(f"Training on {'full' if cfg.train.use_full_data else 'partial'} dataset.")
print(f"Number of training samples: {len(trainset)}")
print(f"Number of validation samples: {len(valset)}")

for epoch_idx in range(cfg.train.epoch_start, cfg.train.epoch_num):
    # train
    train_top1, train_loss, train_lr = train_engine.train_multi_class(train_loader=train_loader, epoch_idx=epoch_idx,
                                                                      ema_start=ema_start)
    # val
    val_top1, val_loss, val_auc = train_engine.val_multi_class(val_loader=val_loader, epoch_idx=epoch_idx)
    # ema_val
    if ema_start:
        ema_val_top1, ema_val_loss, ema_val_auc = train_engine.val_ema(val_loader=val_loader, epoch_idx=epoch_idx)

    train_engine.save_checkpoint(log_root, epoch_idx, train_top1, val_top1, ema_start)

    if ema_start:
        outInfo = f"epoch_idx = {epoch_idx},  train_top1={train_top1}, train_loss={train_loss},val_top1={val_top1},val_loss={val_loss}, val_auc={val_auc}, ema_val_top1={ema_val_top1}, ema_val_loss={ema_val_loss}, ema_val_auc={ema_val_auc} \n"
    else:
        outInfo = f"epoch_idx = {epoch_idx},  train_top1={train_top1}, train_loss={train_loss},val_top1={val_top1},val_loss={val_loss}, val_auc={val_auc} \n"

    print(outInfo)

    f_open.write(outInfo)
    # 刷新文件
    f_open.flush()

    # curve all mAP & mLoss
    writer.add_scalars('top1', {'train': train_top1, 'valid': val_top1}, epoch_idx)
    writer.add_scalars('loss', {'train': train_loss, 'valid': val_loss}, epoch_idx)

    # curve lr
    writer.add_scalar('train_lr', train_lr, epoch_idx)
