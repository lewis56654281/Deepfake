import os
import time
import datetime
import torch
import sys
import psutil

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
cfg.network.name = 'convnext'
cfg.network.class_num = 2
cfg.network.input_size = 384

# : train params
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

cfg.train = CN(new_allowed=True)
cfg.train.resume = True  # 设置为 True 以继续训练
cfg.train.resume_path = '/home/tiancheng/Deepfake/DeepFakeDefenders/output/2024-10-24-23-12-11_convnext_to_competition_BinClass/ema_checkpoint_epoch_7.pth'  # 指定之前保存的模型权重路径
cfg.train.params_path = ''
cfg.train.batch_size = 24
cfg.train.epoch_num = 20
cfg.train.epoch_start = 0
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

# 在训练循环开始前，打印使用的数据量
print(f"Training on {'full' if cfg.train.use_full_data else 'partial'} dataset.")
print(f"Number of training samples: {len(trainset)}")
print(f"Number of validation samples: {len(valset)}")

# 动态设置 worker_num
cpu_count = psutil.cpu_count(logical=False)  # 获取物理CPU核心数
cfg.train.worker_num = min(cpu_count * 2, 8)  # 设置为核心数的2倍，但不超过8

# 动态设置 batch_size
total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
free_gpu_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
suggested_batch_size = free_gpu_memory // (3 * 1024 * 1024 * 1024) * 8  # 假设每个样本大约需要3GB显存，并留有余量
cfg.train.batch_size = max(16, min(suggested_batch_size, 128))  # 设置在16到128之间

print(f"Using worker_num: {cfg.train.worker_num}")
print(f"Using batch_size: {cfg.train.batch_size}")

# 在创建 train_engine 之后添加以下代码
if cfg.train.resume:
    checkpoint = torch.load(cfg.train.resume_path)
    train_engine.netloc_.load_state_dict(checkpoint['model_state_dict'])
    train_engine.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
    cfg.train.epoch_start = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    print(f"Resuming training from epoch {cfg.train.epoch_start}")

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

    # 在每个 epoch 结束时
    checkpoint = {
        'epoch': epoch_idx,
        'model_state_dict': train_engine.netloc_.state_dict(),
        'optimizer_state_dict': train_engine.optimizer_.state_dict(),
        'train_top1': train_top1,
        'val_top1': val_top1,
    }
    torch.save(checkpoint, f"{log_root}/checkpoint_epoch_{epoch_idx}.pth")
