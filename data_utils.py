import os
import torch
from torchvision import datasets, transforms, models
from PIL import Image

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test",
    "chestxray_train":"data/ChestXray/train",
    "chestxray_val":"data/ChestXray/test",
}

# Path to the NIH ChestXray14 metadata CSV (Data_Entry_2017.csv)
CHESTXRAY_CSV = "data/ChestXray/Data_Entry_2017.csv"

CHESTXRAY_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia", "No Finding"
]


class ChestXrayDataset(torch.utils.data.Dataset):
    """Multi-label dataset for NIH ChestXray14.

    Reads images from *img_dir* (searched recursively) and multi-hot labels
    from *csv_path* (Data_Entry_2017.csv).  Returns (image_tensor, label_tensor)
    where label_tensor is a float32 multi-hot vector of length 15.
    """

    def __init__(self, img_dir, csv_path=CHESTXRAY_CSV, transform=None):
        import pandas as pd

        self.transform = transform
        class_to_idx = {c: i for i, c in enumerate(CHESTXRAY_CLASSES)}

        # Build filename -> full path lookup once (supports nested sub-folders)
        fname_to_path = {}
        for root, _, files in os.walk(img_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fname_to_path[fname] = os.path.join(root, fname)

        df = pd.read_csv(csv_path)
        self.img_paths = []
        label_list = []

        for _, row in df.iterrows():
            fname = row['Image Index']
            if fname not in fname_to_path:
                continue
            multi_hot = torch.zeros(len(CHESTXRAY_CLASSES))
            for finding in str(row['Finding Labels']).split('|'):
                finding = finding.strip()
                if finding in class_to_idx:
                    multi_hot[class_to_idx[finding]] = 1.0
            self.img_paths.append(fname_to_path[fname])
            label_list.append(multi_hot)

        self.targets = torch.stack(label_list)  # [N, 15] float32

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

LABEL_FILES = {"places365":"data/categories_places365_clean.txt",
               "imagenet":"data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt",
               "chestxray":"data/chestxray_classes.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in ("chestxray_train", "chestxray_val"):
        data = ChestXrayDataset(DATASET_ROOTS[dataset_name], csv_path=CHESTXRAY_CSV, transform=preprocess)

    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    # ChestXrayDataset exposes .targets as a [N, 15] float tensor (multi-label).
    # ImageFolder / CIFAR expose .targets as a plain list of ints (single-label).
    return pil_data.targets

def get_target_model(target_name, device):
    
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess