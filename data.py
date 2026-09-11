import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.data.distributed import DistributedSampler

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [i for i in os.listdir(root_dir) if i.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_data_loader(args):
    if args.dataset_name == "MNIST":
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)
            ])
        dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        # return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2, 
                                           sampler=DistributedSampler(dataset), drop_last=True)

    if args.dataset_name == "CelebA":
        image_size=64
        dataset = datasets.CelebA('../data', download=True,
                                  transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2, 
                                           sampler=DistributedSampler(dataset), drop_last=True)   

    if args.dataset_name == "CIFAR10":
        image_size=32
        dataset = datasets.CIFAR10('../data', download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2, 
                                           sampler=DistributedSampler(dataset), drop_last=True)   
    
    if args.dataset_name == "CelebA256":
        image_size=256
        dataset = CustomImageDataset(args.data_path, 
                                  transform=transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=2, 
                                           sampler=DistributedSampler(dataset), drop_last=True)   
    
    elif args.dataset_name == "FFHQ":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = CustomImageDataset(args.data_path, transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset), num_workers=2, drop_last=True
        )
        return loader


    elif args.dataset_name == "FFHQ128":
        transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = CustomImageDataset(args.data_path, transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset), num_workers=2, drop_last=True
        )
        return loader
    elif args.dataset_name == 'SVHN':
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)
            ])
        dataset = datasets.SVHN('../data', split='train', transform=transform, download = True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        return loader
    raise ValueError('Unknown data provider')


def get_source_data_loader(args):
    if args.noise_type == "svhn":
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)
            ])
        dataset = datasets.SVHN('../data', split='train', transform=transform, download = True)
        # dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return None
