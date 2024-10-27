import os
import csv
from PIL import Image
import numpy as np
import timm
import einops
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from toolkit.dtransform import create_transforms_inference, create_transforms_inference1,\
                    create_transforms_inference2,\
                    create_transforms_inference3,\
                    create_transforms_inference4,\
                    create_transforms_inference5
from toolkit.chelper import load_model
import torch.nn.functional as F


def extract_model_from_pth(params_path, net_model):
    checkpoint = torch.load(params_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']

    net_model.load_state_dict(state_dict, strict=True)

    return net_model


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        self.kernel = nn.Parameter(torch.from_numpy(self._build_kernel(inc)).float(), requires_grad=False)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        return filters


class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensors = []
        for transform in self.transforms:
            current_data = transform(img)
            img_tensors.append(current_data)
        img_tensor = torch.stack(img_tensors, dim=0)
        return img_tensor, os.path.basename(img_path)


class INFER_API:

    _instance = None
        
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(INFER_API, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.transformer_ = [create_transforms_inference(h=512, w=512),
                        create_transforms_inference1(h=512, w=512),
                        create_transforms_inference2(h=512, w=512),
                        create_transforms_inference3(h=512, w=512),
                        create_transforms_inference4(h=512, w=512),
                        create_transforms_inference5(h=512, w=512)]
        self.srm = SRMConv2d_simple()

        # model init
        self.model = load_model('all', 2 , use_sync_bn=False)
        model_path = '/home/tiancheng/Deepfake/DeepFakeDefenders/final_model_csv/final_model.pth'
        self.model = extract_model_from_pth(model_path, self.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.srm = self.srm.to(self.device)

        self.model.eval()

    def _add_new_channels_worker(self, image):
        # image shape: (batch_size * num_transforms, height, width, channels)
        new_channels = []

        image = einops.rearrange(image, "b h w c -> b c h w")
        image = (image - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)
        srm = self.srm(image)
        new_channels.append(einops.rearrange(srm, "b c h w -> b h w c"))

        new_channels = torch.cat(new_channels, dim=-1)
        return new_channels

    def add_new_channels(self, images):
        # images shape: (batch_size, num_transforms, channels, height, width)
        b, t, c, h, w = images.shape
        images = images.view(b * t, c, h, w)
        images_copied = einops.rearrange(images, "b c h w -> b h w c")
        new_channels = self._add_new_channels_worker(images_copied)
        images_copied = torch.cat([images_copied, new_channels], dim=-1)
        images_copied = einops.rearrange(images_copied, "b h w c -> b c h w")
        return images_copied.view(b, t, -1, h, w)

    def test(self, img_path):
        # img load
        img_data = Image.open(img_path).convert('RGB')

        # transform
        all_data = []
        for transform in self.transformer_:
            current_data = transform(img_data)
            current_data = self.add_new_channels(current_data)
            all_data.append(current_data)
        img_tensor = torch.stack(all_data, dim=0).unsqueeze(0).cuda()

        preds = self.model(img_tensor)

        return round(float(preds), 20)

    def test_batch(self, dataloader):
        all_preds = []
        all_paths = []

        with torch.no_grad():
            for batch, paths in dataloader:
                batch = batch.to(self.device)
                batch = self.add_new_channels(batch)
                b, t, c, h, w = batch.shape
                # 直接使用模型期望的形状
                batch = batch.view(b, t, c, h, w)  # 形状为 (b, t, c, h, w)
                preds = self.model(batch)
                binary_preds = (preds >= 0.5).int().cpu().numpy()
                all_preds.extend(binary_preds.flatten().tolist())
                all_paths.extend(paths)

        return all_preds, all_paths


def process_directory(directory_path, output_file, batch_size=32):
    infer_api = INFER_API()
    
    # 获取目录中所有图片文件
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # 按字典序排序（区分大小写）
    image_files.sort()

    # 准备完整的图片路径列表
    img_paths = [os.path.join(directory_path, f) for f in image_files]

    # 创建数据集和数据加载器
    dataset = ImageDataset(img_paths, infer_api.transformer_)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 批量处理图片
    results, paths = infer_api.test_batch(dataloader)

    # 将结果写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path, result in zip(paths, results):
            # 去除文件扩展名
            filename_without_extension = os.path.splitext(os.path.basename(path))[0]
            writer.writerow([filename_without_extension, result])

    print(f"Results have been saved to {output_file}")


def main():
    input_directory = '/home/tiancheng/Deepfake/DataB/'  # 替换为实际的输入图片目录
    output_file = './results.csv'  # 替换为所需的输出文件路径
    batch_size = 16  # 可以根据您的GPU内存调整这个值
    process_directory(input_directory, output_file, batch_size)
    print(f"Results have been saved to {output_file}")


if __name__ == '__main__':
    main()
