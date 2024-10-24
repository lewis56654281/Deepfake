from toolkit.chelper import final_model
import torch
import os
from collections import OrderedDict


# Trained ConvNeXt and RepLKNet paths (for reference)
convnext_path = '/home/tiancheng/Deepfake/DeepFakeDefenders/output/2024-10-23-23-04-40_convnext_to_competition_BinClass/ema_checkpoint_epoch_19.pth'
replknet_path = '/home/tiancheng/Deepfake/DeepFakeDefenders/output/2024-10-23-11-14-28_replknet_to_competition_BinClass/checkpoint_epoch_19.pth'

model = final_model()

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.model.", "")
        new_state_dict[name] = v
    return new_state_dict

# 加载 ConvNeXt 模型
convnext_data = torch.load(convnext_path, map_location='cpu', weights_only=True)
if 'model_state_dict' in convnext_data:
    state_dict = remove_module_prefix(convnext_data['model_state_dict'])
elif 'ema_model_state_dict' in convnext_data:
    state_dict = remove_module_prefix(convnext_data['ema_model_state_dict'])
else:
    print("Unable to find state_dict in ConvNeXt file. Please check the file structure.")
    print("Available keys:", convnext_data.keys())
    exit(1)

model.convnext.load_state_dict(state_dict, strict=False)
print("ConvNeXt model loaded successfully.")

# 加载 RepLKNet 模型
replknet_data = torch.load(replknet_path, map_location='cpu', weights_only=True)
if 'model_state_dict' in replknet_data:
    state_dict = remove_module_prefix(replknet_data['model_state_dict'])
elif 'ema_model_state_dict' in replknet_data:
    state_dict = remove_module_prefix(replknet_data['ema_model_state_dict'])
else:
    print("Unable to find state_dict in RepLKNet file. Please check the file structure.")
    print("Available keys:", replknet_data.keys())
    exit(1)

model.replknet.load_state_dict(state_dict, strict=False)
print("RepLKNet model loaded successfully.")

# 创建输出目录
if not os.path.exists('./final_model_csv'):
    os.makedirs('./final_model_csv')

# 保存合并后的模型
torch.save({'state_dict': model.state_dict()}, './final_model_csv/final_model.pth')

print("Model merged and saved successfully.")
