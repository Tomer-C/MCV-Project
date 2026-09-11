import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import models
from data import get_data_loader, get_source_data_loader
import argparse
import sys
import os
import wandb
from util import LossManager, prepare_input, log_epoch_save_ckpt, resume, ddp_setup, adjust_learning_rate, update_ema, clamp_d_loss
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_world_size
import time
from contextlib import nullcontext


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    
    # meta
    parser.add_argument('--project_name', default='IGN_MNIST', help='for wandb and directory name')
    parser.add_argument('--run_name', help='for wandb and directory name')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # data
    parser.add_argument('--dataset_name', default='MNIST', help='MNIST, FFHQ, CelebA, ImageNet?, CIFAR10?')
    parser.add_argument("--data_path", type=str, default='/shared/group/ffhq/images/', help="for big datasets that aren't automatically downloaded")
    
    # architecture
    parser.add_argument('--model_name', default='CNN', help='MNIST, VQ, DCGAN, UNet, ViT')
    parser.add_argument('--pretrained_ae', default='None', help='AutoencoderKL, None')
    parser.add_argument('--encode_z', action='store_true', help='whether to encode z as well')
    parser.add_argument('--vit_bottleneck_bn', action='store_true', help='whether to use batch norm in ViT bottleneck')
    parser.add_argument('--mix_zx', action='store_true', help='whether to mix z and x in f')
    parser.add_argument('--mix_zx_pval', default=0.25, type=float, help='probability of mixing z and x for each value')
    parser.add_argument('--mix_zx_pinst', default=0.25, type=float, help='probability of mixing z and x for each instance')
    parser.add_argument('--mix_zx_detach', action='store_true', help='whether to detach x when mixing z and x')
    parser.add_argument('--batch_discrimination', action='store_true', help='whether to use batch discrimination block in vit bottleneck')
    parser.add_argument('--vit_bn', action='store_true', help='whether to use batch norm in vit')

    # optimization
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument("--noise_type", type=str, default='normal', help="normal, uniform")
    parser.add_argument('--num_epochs', default=99999, type=int, help='number of epochs to train')
    parser.add_argument('--optimizer', default='Adam', help='Adam, AdamW')
    parser.add_argument('--base_learning_rate', default=3e-4, type=float, help='base learning rate for optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='learning rate (absolute lr)')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for optimizer')
    parser.add_argument('--timm_weight_decay', default=0., type=float, help='weight decay for optimizer')
    parser.add_argument('--use_consine_scheduler', action='store_true', help='whether to use cosine scheduler')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='number of epochs for linear warmup')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--ema', action='store_true', help='whether to use exponential moving average')
    parser.add_argument('--ema_decay', default=0.9, type=float, help='ema decay')
    parser.add_argument('--use_amp', action='store_true', help='whether to use automatic mixed precision')
    parser.add_argument('--accum_iters', default=1, type=int, help='number of steps to accumulate before updating params')
    parser.add_argument('--epoch_end_xz_interp', default=0, type=int, help='until this epoch z is epoch-linearly interpolated with x')
    parser.add_argument('--grad_clip', action='store_true', help='whether to use gradient clipping')


    parser.add_argument('--conditional', action='store_true', dest='conditional')
    parser.set_defaults(conditional=False)
    
    # devices
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--ddp', action='store_true', help='whether to use multiple gpus. use when in torchrun')


    # logging
    parser.add_argument('--use_wandb', default='cuda', help='whether to log with weights & biases')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--log_freq', default=1, type=int, help='in epochs')
    parser.add_argument('--num_ims_to_show_in_train', default=6, type=int, help='number of epochs to train')
    parser.add_argument('--save_ckpt_freq', default=1, type=int, help='frequency of saving network weights')
    parser.add_argument('--num_seq_f_test', default=4, type=int, help='how many sequential applications f(f(f(...))) in test')


    # losses
    parser.add_argument('--in_manifold_loss_factor', default=1., type=float, help='relative weight of discriminative loss of f')
    parser.add_argument('--out_of_manifold_loss_factor', default=1., type=float, help='relative weight of discriminative loss of f')
    parser.add_argument('--discriminative_loss_factor', default=0.5, type=float, help='relative weight of discriminative loss of f')
    parser.add_argument('--discriminative_loss_clamp_ratio', default=9999., type=float, help='we clamp the discriminative loss to be at most this ratio of the in_manifold_loss')
    parser.add_argument('--discriminative_loss_clamp_val', default=0.5, type=float, help='we clamp the discriminative loss to be at most this ratio of the in_manifold_loss')
    parser.add_argument('--match_means_b_factor', default=0., type=float, help='relative weight of matching means of f(x) and f(z) (only for bottleneck)')
    parser.add_argument('--match_stds_b_factor', default=0., type=float, help='relative weight of matching stds of f(x) and f(z) (only for bottleneck)')
    parser.add_argument('--match_means_o_factor', default=0., type=float, help='relative weight of matching means of f(x) and f(z) (only for outer layer)')
    parser.add_argument('--match_stds_o_factor', default=0., type=float, help='relative weight of matching stds of f(x) and f(z) (only for outer layer)')
    parser.add_argument('--epoch_begin_idempotence_losses', default=0, type=int, help='until this epoch it is only autoencoding')
    parser.add_argument('--loss_p', default=2., type=float, help='power of loss, typically 1.0 or 2.0')

    # generated input instead of noise
    parser.add_argument('--prob_input_is_prev_gen', default=0.5, type=float, 
                        help='this is probability for taking prev generated images as input instead of noise. works momentum style f(f(f(...)')
    parser.add_argument('--epoch_begin_gen_input', default=9999999, type=int, 
                        help='from this epoch z is replaced with previously generated with given probability')
    
    return parser


def train(f, f_copy, data_loader, pretrained_e, pretrained_d, args, rank, source_data_loader):
    l = LossManager(f, args)
    lr = args.lr
    f_z, fx_b = None, None
    start_epoch = resume(f, l, args)
    
    for epoch in range(start_epoch, args.num_epochs):
        if args.ddp:
            data_loader.sampler.set_epoch(epoch)
        if source_data_loader is not None:
            source_data_loader_iter = iter(source_data_loader)
        else:
            source_data_loader_iter = None
        # data_loader.sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(data_loader):
            if args.use_consine_scheduler:
                lr = adjust_learning_rate(l.opt, batch_idx / len(data_loader) + epoch, args)

            # x is data, z is noise or prev generated
            x, z, x_raw, z_raw, pure_z, cond = prepare_input(data, epoch, f_z, pretrained_e, args, rank, source_data_loader_iter)
            
            # update f_copy
            if args.ema:
                update_ema(f_copy, f, args.ema_decay)
            else:
                f_copy.load_state_dict(f.state_dict())

            # apply f to get all needed outputs
            with torch.cuda.amp.autocast() if args.use_amp else nullcontext():
                fx = f(x, cond=cond)
                fz = f(z, cond=cond)
                # with torch.no_grad():
                # f_z = f_copy(z, mix_zx=False)
                if isinstance(fx, tuple):
                    (fx, fx_b), (fz, fz_b) = fx, fz
                    # (fx, fx_b), (fz, fz_b), (f_z, _) = fx, fz, f_z
                # f_z = f_z.detach()
                f_z = fz.detach()
                f_fz = f_copy(fz, cond=cond, mix_zx=False)
                ff_z = f(f_z, cond=cond, mix_zx=False)
                if isinstance(f_fz, tuple):
                    (f_fz, _), (ff_z, _) = f_fz, ff_z

                # calculate losses
                l.add_loss("loss_in", (fx - x.detach()).abs().pow(args.loss_p).mean())  # f(x)=x
                l.add_loss("loss_out", (f_fz - fz).abs().pow(args.loss_p).mean())  # f(f(z))=f(z) optimize inner f
                l.add_loss("loss_out_d", -clamp_d_loss((ff_z - f_z).abs().pow(args.loss_p), l).mean())  # f(f(z))!=f(z) optimize outer f, only for f(f(z)) close to f(z)
                if fx_b is not None:  # match bottleneck stats losses only if bottleneck was outputed
                    l.add_loss("match_means_b", (fx_b.mean(0).detach() - fz_b.mean(0)).abs().pow(args.loss_p).mean()) 
                    l.add_loss("match_stds_b", (fx_b.std(0).detach() - fz_b.std(0)).abs().pow(args.loss_p).mean())
                l.add_loss("match_means_o", (fx.mean(0).detach() - fz.mean(0)).abs().pow(args.loss_p).mean()) 
                l.add_loss("match_stds_o", (fx.std(0).detach() - fz.std(0)).abs().pow(args.loss_p).mean())

                # optimize for losses
                l.agg_zero_back_step(z.shape[0], epoch, batch_idx)

        #log losses and images, save checkpoint
        test_exmaples = test(f, pure_z, cond, rank, args)
        log_epoch_save_ckpt(args, l, f, x, x_raw, fx, z, z_raw, fz, f_fz, test_exmaples, lr, pretrained_d, epoch, rank)


@torch.no_grad()
def test(f, z, cond, rank, args):
    # if rank == 0:
    with torch.cuda.amp.autocast() if args.use_amp else nullcontext():
        out = [z[:args.num_ims_to_show_in_train]]
        for _ in range(args.num_seq_f_test):
            # TODO(yossi): is the condtioning here correct?
            if cond is not None:
                cur_out = f(out[-1], cond=cond[:args.num_ims_to_show_in_train])
            else:
                cur_out = f(out[-1])
            if isinstance(cur_out, tuple):
                cur_out = cur_out[0]
            out.append(cur_out)
    return out[1:]


def main(args):
    # prepare distributed
    # torch.autograd.set_detect_anomaly(True)
    if args.ddp:
        rank = ddp_setup()
    else:
        rank = 0

    # prepare wandb and checkpoint dir
    if rank==0:
        print(args)
        if args.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.project_name, entity="assafshocher", config=vars(args))
                wandb.run.name = args.run_name
            except Exception as e:
                print(f"Unable to setup wandb: {e}")
        args.dir = '%s_checkpoint/%s_%s' % (args.project_name, args.run_name, time.strftime("%m%d_%H%M%S"))
        os.makedirs(args.dir, exist_ok=True)

    # prepare learning rate
    if args.lr is None:  # only base_lr is specified
        world_size = get_world_size() if args.ddp else 1
        eff_batch_size = args.batch_size * world_size * args.accum_iters
        args.lr = args.base_learning_rate * eff_batch_size / 256
    
    # prepare data
    data_loader = get_data_loader(args)
    source_data_loader = get_source_data_loader(args)
    # prepare model
    f, f_copy, pretrained_e, pretrained_d = models.get_model(args, rank)

    # all set, start training
    train(f, f_copy, data_loader, pretrained_e, pretrained_d, args, rank, source_data_loader)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)