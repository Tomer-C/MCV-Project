"""IGN architectures, adapted from Shocher et al.'s IGN-main/models.

Two changes from upstream:

  * `nn.SyncBatchNorm` is replaced by a `norm_layer` factory that resolves to
    SyncBatchNorm only when DDP is actually initialised, and to plain
    BatchNorm2d otherwise. Upstream crashes outside torchrun because
    SyncBatchNorm requires a process group.
  * `forward` returns a single tensor rather than the (image, bottleneck)
    tuple, since we do not use the bottleneck-statistics losses. The
    bottleneck is still exposed via `forward_with_bottleneck` for anyone who
    wants to re-enable them.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def resolve_norm(kind="auto"):
    """Pick a normalisation layer that is valid for the current process setup."""
    if kind == "batch":
        return nn.BatchNorm2d
    if kind == "instance":
        return lambda c, **kw: nn.InstanceNorm2d(c, affine=True)
    if kind == "none":
        return lambda c, **kw: nn.Identity()
    # auto: SyncBatchNorm is only legal once a process group exists
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return nn.SyncBatchNorm
    return nn.BatchNorm2d


def _norm(norm_layer, channels):
    """SyncBatchNorm/BatchNorm2d take `affine`; Identity/InstanceNorm wrapper does not."""
    try:
        return norm_layer(channels, affine=True)
    except TypeError:
        return norm_layer(channels)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, num_convs=2):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            _norm(norm_layer, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        block = []
        for _ in range(num_convs - 1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            block.append(nn.LeakyReLU(0.2, inplace=True))
        self.convs = nn.Sequential(*block)

    def forward(self, x):
        x = self.down_conv(x)
        return self.convs(x) + x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, num_convs=2, stride=2):
        super().__init__()
        padding = 1 if stride == 2 else 0
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, padding, bias=False),
            _norm(norm_layer, out_channels),
            nn.ReLU(True),
        )
        block = []
        for _ in range(num_convs - 1):
            block.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            block.append(_norm(norm_layer, out_channels))
            block.append(nn.ReLU(True))
        self.convs = nn.Sequential(*block)

    def forward(self, x):
        x = self.up_conv(x)
        return self.convs(x) + x


class DCGANEnhanced(nn.Module):
    """Residual 64x64 encoder-decoder from IGN-main/models/dcgan_enhanced.py.

    NOT the architecture the paper describes (that is DCGANPlain below); this
    is the skeleton's enhanced variant with extra 3x3 convs and residual skips.
    Input and output are both (B, 3, 64, 64) in [-1, 1], which is what makes
    f composable with itself -- and, for this project, composable with a
    *different* IGN f_B.
    """

    def __init__(self, norm="auto", ngf=64, ndf=64, nc=3):
        super().__init__()
        norm_layer = resolve_norm(norm)

        self.down_blocks = nn.Sequential(
            DownBlock(nc, ndf, norm_layer, num_convs=2),            # 64 -> 32
            DownBlock(ndf, ndf * 2, norm_layer, num_convs=2),       # 32 -> 16
            DownBlock(ndf * 2, ndf * 4, norm_layer, num_convs=2),   # 16 -> 8
            DownBlock(ndf * 4, ndf * 8, norm_layer, num_convs=2),   # 8  -> 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),       # 4  -> 1
        )
        self.up_blocks = nn.Sequential(
            UpBlock(ndf * 8, ngf * 8, norm_layer, num_convs=2, stride=1),  # 1 -> 4
            UpBlock(ngf * 8, ngf * 4, norm_layer, num_convs=2),            # 4 -> 8
            UpBlock(ngf * 4, ngf * 2, norm_layer, num_convs=2),            # 8 -> 16
            UpBlock(ngf * 2, ngf, norm_layer, num_convs=2),                # 16 -> 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),              # 32 -> 64
            nn.Tanh(),
        )
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Norm") != -1 and getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward_with_bottleneck(self, x):
        b = self.down_blocks(x)
        return self.up_blocks(b), b

    def forward(self, x):
        return self.up_blocks(self.down_blocks(x))


class DCGANPlain(nn.Module):
    """The 5-layer DCGAN autoencoder the paper actually describes (Appendix C).

    Encoder = DCGAN discriminator backbone (no norm after conv1 or conv5),
    decoder = DCGAN generator. No residual blocks and no extra 3x3 convs --
    those belong to the 'enhanced' variant. Note the skeleton's own
    IGN-main/models/dcgan.py is stale (its decoder expects 256 input channels
    but receives 512, so it cannot run); this class is built from the paper's
    architecture table instead.
    """

    def __init__(self, norm="auto", ngf=64, ndf=64, nc=3):
        super().__init__()
        norm_layer = resolve_norm(norm)

        self.down_blocks = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),            # 64 -> 32, no norm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),       # 32 -> 16
            _norm(norm_layer, ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),   # 16 -> 8
            _norm(norm_layer, ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),   # 8 -> 4
            _norm(norm_layer, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),   # 4 -> 1, no norm/act
        )
        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ngf * 8, 4, 1, 0, bias=False),  # 1 -> 4
            _norm(norm_layer, ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 4 -> 8
            _norm(norm_layer, ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8 -> 16
            _norm(norm_layer, ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 16 -> 32
            _norm(norm_layer, ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # 32 -> 64
            nn.Tanh(),
        )
        self.apply(DCGANEnhanced.weights_init)

    def forward_with_bottleneck(self, x):
        b = self.down_blocks(x)
        return self.up_blocks(b), b

    def forward(self, x):
        return self.up_blocks(self.down_blocks(x))


ARCHS = {"dcgan": DCGANPlain, "dcgan_enhanced": DCGANEnhanced}


class AttributeClassifier(nn.Module):
    """Small CNN that predicts CelebA attributes from a 64x64 image.

    This is the evaluation instrument, not part of the IGN. It answers the
    question the proposal actually cares about: does an image produced by
    alternating f_A and f_B genuinely carry *both* attributes?
    """

    def __init__(self, num_attrs=40, ndf=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(ndf * 8 * 4 * 4, num_attrs))

    def forward(self, x):
        return self.head(self.features(x))  # logits


def build_ign(args, device):
    """Instantiate f and its frozen copy f_copy, mirroring upstream's setup."""
    norm = getattr(args, "norm", "auto")
    arch = ARCHS[getattr(args, "arch", "dcgan")]
    f = arch(norm=norm).to(device)
    f_copy = arch(norm=norm).to(device)
    f_copy.load_state_dict(f.state_dict())

    if getattr(args, "ddp", False) and dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        f = DDP(f, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False)
        f_copy = DDP(f_copy, device_ids=[device.index] if device.type == "cuda" else None,
                     broadcast_buffers=False)
    f.train()
    f_copy.train()
    return f, f_copy


def unwrap(model):
    """Strip DDP so state dicts are saved/loaded consistently across setups."""
    return model.module if hasattr(model, "module") else model


def load_ign(ckpt_path, device, norm="batch"):
    """Load a trained IGN for inference. Used by every experiment script.

    The architecture is read from the checkpoint's saved args; checkpoints from
    before the --arch flag existed were all DCGANEnhanced, so that is the
    fallback.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ARCHS[ckpt.get("args", {}).get("arch", "dcgan_enhanced")]
    model = arch(norm=norm).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AttributeClassifier().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
