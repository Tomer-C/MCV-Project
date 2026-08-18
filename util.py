import torch
import wandb
import torch.optim as optim
from torch.distributed import init_process_group
import os
import torch.distributed as dist
import numpy as np
import math
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur


def ddp_setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"Start running on rank {rank}.")
    return rank


class LossManager:
    @torch.no_grad()
    def __init__(self, f, args):
        # print("LossManager init", dist.get_rank())
        self.loss = 0
        self.loss_dict = {}
        self.count = 0
        self.cur_loss_dict = {}
        self.weight_dict = {"loss_in": args.in_manifold_loss_factor, 
                            "loss_out": args.out_of_manifold_loss_factor, 
                            "loss_out_d": args.discriminative_loss_factor,
                            "match_means_b": args.match_means_b_factor,
                            "match_stds_b": args.match_stds_b_factor,
                            "match_means_o": args.match_means_o_factor,
                            "match_stds_o": args.match_stds_o_factor,}

        self.loss_p = args.loss_p

        if args.timm_weight_decay > 0:
            import timm.optim.optim_factory as optim_factory
            param_groups = optim_factory.param_groups_weight_decay(f.module, args.timm_weight_decay)
        else:
            param_groups = f.parameters()
        if args.optimizer == "Adam":
            self.opt = optim.Adam(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optimizer == "AdamW":
            self.opt = optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))

        self.discriminative_loss_clamp_ratio = args.discriminative_loss_clamp_ratio
        self.discriminative_loss_clamp_val = args.discriminative_loss_clamp_val
        self.epoch_begin_idempotence_losses = args.epoch_begin_idempotence_losses
        
        self._scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

        self.accum_iters = args.accum_iters

        self.f = f
        self.grad_clip = args.grad_clip

    def get_final_loss(self, loss_dict, epoch):
        if epoch < self.epoch_begin_idempotence_losses:
            return loss_dict["loss_in"] * self.weight_dict["loss_in"]
        loss = 0
        for k, v in loss_dict.items():
            loss = loss + v * self.weight_dict[k]
        return loss

    def add_loss(self, l_name, l_value):
    # def add_loss(self, l_name, unaveraged_l_val, clamp=False, negate=False):
        # if 'loss_in' in self.cur_loss_dict:
        #     # if not, it means now processing loss_in. no use of x0 will be made
        #     x0 = self.cur_loss_dict['loss_in'].detach()

        # # unaveraged_loss_Val is expected to be some diff that can be negative
        # x = unaveraged_l_val.abs()

        # if negate:
        #     x = (2 * x0 - x).relu()
        
        # x = x ** self.loss_p
        
        # x = x.view(x.shape[0], -1).mean(1) # average all dims but batch

        # if clamp:
        #     x = F.elu((x - x0) / x0, alpha=0.1) * x0 + x0  # +x0 doesn nothing to grads, but easier to comp with loss_in

        # l_value = x.mean()
        if l_name not in self.cur_loss_dict:
            self.cur_loss_dict[l_name] = 0
        self.cur_loss_dict[l_name] += l_value

    @torch.no_grad()
    def agg_losses_avg(self, cur_batch_size):
        self.loss_dict = {k: v*self.count for k, v in self.loss_dict.items()}
        for k, v in self.cur_loss_dict.items():
            if k not in self.loss_dict:
                self.loss_dict[k] = 0
            self.loss_dict[k] += v.item() * cur_batch_size
        self.count += cur_batch_size
        self.loss_dict = {k: v/self.count for k, v in self.loss_dict.items()}

    @torch.no_grad()
    def update_weights(self, weight_dict):
        self.weight_dict.update(weight_dict)

    @torch.no_grad()
    def reset(self):
        self.loss_dict = {}
        self.count = 0

    @torch.no_grad()
    def __str__(self):
        return str(self.loss_dict)

    @torch.no_grad()
    def log_and_reset(self, epoch, log_to_wandb=True):
        if log_to_wandb:
            final_loss = self.get_final_loss(self.loss_dict, epoch)
            self.loss_dict["final_loss"] = final_loss
            self.loss_dict["epoch"] = epoch
            wandb.log(self.loss_dict, step=epoch)
        print(epoch, self)
        self.reset()

    def agg_zero_back_step(self, cur_batch_size, epoch, batch_idx):
        self.agg_losses_avg(cur_batch_size)
        loss = self.get_final_loss(self.cur_loss_dict, epoch) / self.accum_iters
        self.cur_loss_dict = {}
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.f.parameters(), 0.1)
        if batch_idx % self.accum_iters == 0:
            if self._scaler is not None:
                self._scaler.unscale_(self.opt)  # unscale the gradients of optimizer's assigned params in-place
                self._scaler.step(self.opt)
                self._scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad()
    

@torch.no_grad()
def log_epoch_save_ckpt(args, loss_manager, f, x, x_raw, fx, z, z_raw, fz, f_fz, test_examples, lr, pretrained_d, epoch, rank):
    if rank == 0:
        if args.use_wandb:
            d = pretrained_d
            n = args.num_ims_to_show_in_train
            images = {"x": wandb.Image(x_raw[:n]),
                      "f(x)": wandb.Image(d(fx[:n])),
                      "f(z)": wandb.Image(d(fz[:n])),
                      "f(f(z))": wandb.Image(d(f_fz[:n])),
                      "d(e(x))": wandb.Image(d(x[:n]))}
            if args.encode_z:
                images["z"] = wandb.Image(z_raw[:n])
                images["d(e(z))"] = wandb.Image(d(z[:n]))
            else:
                images["d(z)"] = wandb.Image(d(z[:n]))

            if test_examples is not None:
                title = "z"
                for im in test_examples:
                    title = "f(%s)" % title
                    images["test %s" % title] = wandb.Image(d(im[:n]))

            wandb.log(images, step=epoch)
            wandb.log({'lr': lr}, step=epoch)

        if epoch % args.log_freq == 0:
            loss_manager.log_and_reset(epoch, log_to_wandb=args.use_wandb)
        if epoch % args.save_ckpt_freq == 0:
            model = f.module if args.ddp else f
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': loss_manager.opt.state_dict(),
                        }, f"%s/ckpt_%d.pt" % (args.dir, epoch))
            if args.use_wandb:
                wandb.save(f"%s/ckpt_%d.pt" % (args.dir, epoch))
            # delete old checkpoints according to a policy
            clean_checkpoints(epoch, args)

def resume(f, loss_manager, args):
    if not args.resume:
        return 0
    if loss_manager is not None:
        rank = dist.get_rank()
    else:
        rank = args.device
    checkpoint = torch.load(args.resume, map_location=torch.device(rank))
    state_dict = checkpoint['model_state_dict']
    if not 'module' in list(state_dict.keys())[0] and loss_manager is not None:
        state_dict = {"module.%s" %k: v for k, v in checkpoint['model_state_dict'].items()}
    f.load_state_dict(state_dict)
    if loss_manager is not None:
        loss_manager.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"resuming from epoch {epoch}")
    if args.use_wandb and rank == 0:
        wandb.restore(args.resume)
    return epoch + 1


def prepare_input(data, epoch, f_z, pretrained_e, args, rank, source_tokens):
    cond = None
    if args.conditional:
        cond = data[1]
        data = data[0]
    if isinstance(data, tuple) or isinstance(data, list):
        data = data[0]
    bsz = data.shape[0]

    # arange input noises, real ims and prev generated ims
    x_raw = data.to(rank)
    x = pretrained_e(x_raw)
    if args.encode_z:
        if args.noise_type == 'normal':
            z_raw = torch.randn_like(x_raw)
        elif args.noise_type == 'uniform':
            z_raw = torch.rand_like(x_raw)
        elif args.noise_type == 'smooth':
            z_raw = torch.randn_like(x_raw)
            z_raw = gaussian_blur(z_raw, 7)
        elif args.noise_type == 'svhn':
            z_raw, _ = next(source_tokens)
            z_raw = z_raw.to(x.get_device())
        z = pretrained_e(z)
    else:
        if args.noise_type == 'normal':
            z = torch.randn_like(x)
        elif args.noise_type == 'uniform':
            z = torch.rand_like(x)
        elif args.noise_type == 'smooth':
            z = torch.randn_like(x_raw)
            z = gaussian_blur(z, 27)
        elif args.noise_type == 'x_stats':
            z = torch.randn_like(x)
            z = z * x.std(dim=0) + x.mean(dim=0)
        elif args.noise_type == 'x_fft_stats':
            z = generate_noise_with_image_stats(x)
        elif args.noise_type == 'svhn':
            z, _ = next(source_tokens)
            z = z.to(x.get_device())
        z_raw = None

    pure_z = z

    if epoch < args.epoch_end_xz_interp:
        alpha = epoch / args.epoch_end_xz_interp
        z = alpha * z + (1-alpha) * x

    if epoch > args.epoch_begin_gen_input:
        r = (torch.rand(bsz, device=rank) > args.prob_input_is_prev_gen).float()[:, None, None, None]
        r.to(z.get_device())
        z = r * z + (1-r) * f_z
    return x, z, x_raw, z_raw, pure_z, cond


def clamp_d_loss(raw_loss, l):
    x0_val = l.discriminative_loss_clamp_val
    x0_ratio = (l.cur_loss_dict['loss_in'].detach() * l.discriminative_loss_clamp_ratio)
    x0 = min(x0_val, x0_ratio)
    x = raw_loss.mean((1,2,3))
    return (x / x0).tanh() * x0


def mode_separator(fz, fx, rank):
    B, C, H, W = fz.shape
    fz = fz.view(B, C, H*W)
    fx = fx.view(B, C, H*W)
    fz = F.normalize(fz, dim=1)
    fx = F.normalize(fx, dim=1)
    mat = torch.einsum('bcl,ncl->cbn', fx, fz).softmax(1) # (C, B, N), normalized
    maxes, _ = mat.max(-1)
    return 1 - maxes.mean()


def clean_checkpoints(cur_epoch, args):
    # get all checkpoint files
    checkpoint_files = [file for file in os.listdir(args.dir) if file.startswith('ckpt_') and file.endswith('.pt')]

    for file in checkpoint_files:
        ckpt_epoch = int(file.split('_')[1].split('.')[0])

        if ckpt_epoch > cur_epoch - 10:
            # rule 1: keep the last 10 checkpoints
            continue
        elif ckpt_epoch > cur_epoch - 100 and ckpt_epoch % 10 == 0:
            # rule 2: for the last 100 checkpoints, keep all those divided by 10
            continue
        elif ckpt_epoch > cur_epoch - 200 and ckpt_epoch % 20 == 0:
            # rule 3: for the last 200 checkpoints keep all those divided by 20
            continue
        elif ckpt_epoch <= cur_epoch - 200 and ckpt_epoch % 50 == 0:
            # rule 4: for all the rest checkpoints keep all those divided by 50
            continue
        else:
            # if none of the rules apply, delete the checkpoint
            os.remove(os.path.join(args.dir, file))


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



class SyncBatchNormViT(nn.Module):
    def __init__(self, num_features, eps=1e-6, momentum=0.1, affine=True, track_running_stats=True, rank=None):
        super().__init__()
        self.bn = nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats, rank)
        # self.ln = nn.LayerNorm(num_features, eps, elementwise_affine=affine)

    def forward(self, x):
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        # x = self.ln(x)
        return x

def generate_noise_with_image_stats(images):
    B, C, H, W = images.shape

    # Compute the Fourier transform
    f_images = torch.fft.fftn(images, dim=(-2, -1))
    # print("real000", f_images[0, 0, 0].imag)

    
    # Compute mean and variance for both real and imaginary parts
    real_mean = torch.mean(f_images.real, dim=0, keepdim=True).expand(B, C, H, W)
    real_variance = torch.var(f_images.real, dim=0, keepdim=True).expand(B, C, H, W)
    imag_mean = torch.mean(f_images.imag, dim=0, keepdim=True).expand(B, C, H, W)
    imag_variance = torch.var(f_images.imag, dim=0, keepdim=True).expand(B, C, H, W)
    
    # Generate Gaussian noise for the top-left quarter of the frequency components
    quarterH = H // 2 + H % 2
    quarterW = W // 2 + W % 2

    real_noise_quarter = torch.normal(real_mean[:, :, :quarterH, :quarterW], torch.sqrt(real_variance[:, :, :quarterH, :quarterW]))
    imag_noise_quarter = torch.normal(imag_mean[:, :, :quarterH, :quarterW], torch.sqrt(imag_variance[:, :, :quarterH, :quarterW]))

    # Handle the DC component
    imag_noise_quarter[:, :, 0, 0] = 0

    # Handle symmetry along width to create the top-half
    real_noise_half = torch.cat((real_noise_quarter, real_noise_quarter[..., W % 2:].flip(-1)), dim=-1)
    imag_noise_half = torch.cat((imag_noise_quarter, -imag_noise_quarter[..., W % 2:].flip(-1)), dim=-1)

    # Handle symmetry along height to create the full noise
    real_noise = torch.cat((real_noise_half, real_noise_half[:, :, H % 2:, :].flip(-2)), dim=-2)
    imag_noise = torch.cat((imag_noise_half, -imag_noise_half[:, :, H % 2:, :].flip(-2)), dim=-2)

    # Ensure the midlines are real for even dimensions
    if H % 2 == 0:  # even height
        imag_noise[:, :, H//2, 0] = 0  # Nyquist frequency for height
    if W % 2 == 0:  # even width
        imag_noise[:, :, 0, W//2] = 0  # Nyquist frequency for width


    complex_noise = torch.complex(real_noise, imag_noise)

    # print("noise000", complex_noise[0, 0, 0].imag)


    # Inverse Fourier transform to get spatial representation
    noise_images = torch.fft.ifftn(complex_noise, dim=(-2, -1))
    # sanity = torch.fft.ifftn(f_images, dim=(-2, -1))

    # print('noise ifft:', noise_images[0].imag)
    # print('sanity ifft:', sanity[0].imag)

    # print(noise_images.real.shape, noise_images[0])


    return noise_images.real
