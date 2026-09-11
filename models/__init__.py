import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import torch.nn as nn


def get_model(args, rank, verbose=True):
    if args.model_name == "MNIST":
        from models.mnist import MNIST
        f, f_copy = MNIST(), MNIST()

    if args.model_name == "CNN":
        from models.simple_cnn import CNN
        f, f_copy = CNN(), CNN()

    
    if args.model_name == "MNIST_UNet":
        from models.mnist_unet import MNIST_UNet
        f, f_copy = MNIST_UNet(), MNIST_UNet()
    
    if args.model_name == "CIFAR10":
        from models.cifar10 import CIFAR10
        f, f_copy = CIFAR10(args.conditional), CIFAR10(args.conditional)
    
    elif args.model_name == "VQ":
        from models.vqvae import VQVAE
        f, f_copy = VQVAE(), VQVAE()
    
    elif args.model_name == "DCGAN":
        from models.dcgan import DCGAN
        f, f_copy = DCGAN(), DCGAN()

    elif args.model_name == "DCGAN_nonorm":
        from models.dcgan import DCGAN
        f, f_copy = (DCGAN(norm=nn.Identity), 
                     DCGAN(norm=nn.Identity))

    elif args.model_name == "DCGAN_enhanced":
        from models.dcgan_enhanced import DCGAN_enhanched
        f, f_copy = DCGAN_enhanched(), DCGAN_enhanched()
    
    elif args.model_name == "DCGAN_instnorm":
        from models.dcgan import DCGAN
        f, f_copy = (DCGAN(norm=nn.InstanceNorm2d), 
                     DCGAN(norm=nn.InstanceNorm2d))
    
    elif args.model_name == "UNet":
        from models.unet import UNet
        f, f_copy = UNet(), UNet()

    elif args.model_name == "MUNet":
        from models.unet_mul import MUNet
        f, f_copy = MUNet(), MUNet()

    elif args.model_name == "TinyViTNoBottleNeck":
        from models.bottleneck_vit import BottleNeckViT
        f, f_copy = (BottleNeckViT(), 
                     BottleNeckViT())

    elif "vit_" in args.model_name:
        import models.vit as vit
        if args.vit_bn:
            from util import SyncBatchNormViT
        vit_args = {'bottleneck_bn': args.vit_bottleneck_bn,
                    'mix_zx': args.mix_zx,
                    'mix_p_value': args.mix_zx_pval,
                    'mix_p_instance': args.mix_zx_pinst,
                    'mix_zx_detach': args.mix_zx_detach,
                    'batch_discrimination': args.batch_discrimination,
                    'norm_layer': SyncBatchNormViT if args.vit_bn else partial(nn.LayerNorm, eps=1e-6)}

        f, f_copy = (vit.__dict__[args.model_name](**vit_args), 
                     vit.__dict__[args.model_name](**vit_args))

    elif args.model_name == "StyleGAN":
        from models.stylegan import StyleGANAutoencoder
        f, f_copy = StyleGANAutoencoder(), StyleGANAutoencoder()
    
    if args.ddp:
        f = DDP(f.to(rank), device_ids=[rank], broadcast_buffers=False)
        f_copy = DDP(f_copy.to(rank), device_ids=[rank], broadcast_buffers=False)
    else:
        rank = args.device
        f = f.to(rank)
        f_copy = f_copy.to(rank)

    
    f.train()
    f_copy.train()

    if args.pretrained_ae == "AutoencoderKL":
        from diffusers.models import AutoencoderKL
        pretrained_ae = AutoencoderKL().from_pretrained(f"stabilityai/sd-vae-ft-ema").to(rank)
        pretrained_ae.eval()
        def pretrained_e(x):
            with torch.no_grad():
                return pretrained_ae.encode(x).latent_dist.sample().mul_(0.18215)
        def pretrained_d(x):
            with torch.no_grad():
                return pretrained_ae.decode(x.float() / 0.18215).sample.detach()
    else:
        pretrained_e = torch.nn.Identity()
        pretrained_d = torch.nn.Identity()
        print('***** No valid pretrained_ae specified, using identity function')

    if (rank==0 or rank is None) and verbose:
        print(f)
    return f, f_copy, pretrained_e, pretrained_d

