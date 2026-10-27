"""Attribute-filtered CelebA loaders for the alternating-projection IGN project.

The upstream IGN `data.py` returns the full dataset and always wraps it in a
DistributedSampler. We need two things it does not provide:

  1. Subsets filtered by CelebA attribute predicates, so we can build the
     "A", "B" and "A and B" training sets referenced in the proposal.
  2. Loaders that work without torch.distributed initialised, so the same code
     runs on a laptop, one GPU, or a torchrun cluster.
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# CelebA's 40 binary attributes, in the order torchvision returns them.
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]
ATTR_TO_IDX = {name: i for i, name in enumerate(CELEBA_ATTRS)}


def parse_attr_spec(spec):
    """Parse an attribute spec string into a list of (index, wanted_value).

    Accepts comma-separated attribute names, each optionally prefixed with '-'
    to require the attribute be absent:

        "Smiling"                -> smiling == 1
        "Smiling,Eyeglasses"     -> smiling == 1 AND eyeglasses == 1
        "Smiling,-Male"          -> smiling == 1 AND male == 0
    """
    conditions = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        want = 1
        if raw.startswith(("-", "~")):  # '~' avoids argparse eating a leading '-'
            want, raw = 0, raw[1:].strip()
        if raw not in ATTR_TO_IDX:
            raise ValueError(
                f"Unknown CelebA attribute {raw!r}. Valid names: {', '.join(CELEBA_ATTRS)}"
            )
        conditions.append((ATTR_TO_IDX[raw], want))
    if not conditions:
        raise ValueError(f"Empty attribute spec: {spec!r}")
    return conditions


def spec_to_slug(spec):
    """Filesystem-safe short name for a spec, used for run/checkpoint dirs."""
    return (spec.replace(",", "_AND_").replace("-", "not_")
            .replace("~", "not_").replace(" ", ""))


def celeba_transform(image_size):
    """Match the upstream IGN CelebA preprocessing: center crop to [-1, 1]."""
    return transforms.Compose([
        transforms.CenterCrop(178),   # crop the aligned face before resizing
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class AttrImageDataset(Dataset):
    """Wraps a CelebA subset to return (image, attr_vector) with float attrs.

    Returning the attribute vector (rather than dropping it as upstream does)
    lets the same dataset feed both IGN training and the attribute classifier
    used for evaluation.
    """

    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, attrs = self.base[self.indices[i]]
        return img, attrs.float()


def _celeba_files_present(root):
    """True if a manually-placed CelebA is already at <root>/celeba/."""
    base = os.path.join(root, "celeba")
    needed = ["list_attr_celeba.txt", "list_eval_partition.txt", "identity_CelebA.txt",
              "list_bbox_celeba.txt", "list_landmarks_align_celeba.txt"]
    return (all(os.path.isfile(os.path.join(base, f)) for f in needed)
            and os.path.isdir(os.path.join(base, "img_align_celeba")))


def _load_celeba(root, split, image_size):
    # If the data was placed manually (e.g. via scripts/setup_celeba.py), skip
    # torchvision's md5 gate: converted/mirrored files won't match the official
    # hashes, which would otherwise force a Google-Drive re-download that is
    # frequently rate-limited. Only bypasses the check when the files exist.
    if _celeba_files_present(root):
        datasets.CelebA._check_integrity = lambda self: True
        return datasets.CelebA(
            root, split=split, target_type="attr", download=False,
            transform=celeba_transform(image_size),
        )
    try:
        return datasets.CelebA(
            root, split=split, target_type="attr", download=True,
            transform=celeba_transform(image_size),
        )
    except Exception as e:
        raise RuntimeError(
            "Could not load CelebA. torchvision's Google Drive download is "
            "frequently rate-limited; download the dataset manually and place it at\n"
            f"  {os.path.join(root, 'celeba')}/\n"
            "with img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt, "
            "identity_CelebA.txt, list_bbox_celeba.txt, list_landmarks_align_celeba.txt\n"
            "See scripts/README_data.md for a kaggle one-liner.\n"
            f"Original error: {e}"
        ) from e


def get_attr_indices(base, spec):
    """Indices of samples in `base` satisfying every condition in `spec`."""
    conditions = parse_attr_spec(spec)
    attrs = base.attr  # (N, 40) int64 tensor of 0/1
    mask = torch.ones(len(attrs), dtype=torch.bool)
    for idx, want in conditions:
        mask &= (attrs[:, idx] == want)
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def get_filtered_dataset(args, spec, split="train"):
    """CelebA subset containing only images satisfying `spec`."""
    base = _load_celeba(args.data_root, split, args.image_size)
    if spec is None or spec.lower() in ("all", "none", ""):
        indices = range(len(base))
    else:
        indices = get_attr_indices(base, spec)
    if len(indices) == 0:
        raise ValueError(f"Attribute spec {spec!r} matched 0 images in split {split!r}.")
    return AttrImageDataset(base, indices)


def make_loader(dataset, batch_size, ddp, shuffle=True, num_workers=2, drop_last=True):
    """Build a DataLoader that works with or without torch.distributed.

    Upstream always passes a DistributedSampler, which raises if the process
    group is not initialised. We only use it when DDP is actually active.
    """
    sampler = None
    if ddp and dist.is_available() and dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, drop_last=drop_last, pin_memory=torch.cuda.is_available(),
    )


def get_data_loader(args, spec, split="train", shuffle=True, drop_last=True):
    dataset = get_filtered_dataset(args, spec, split)
    loader = make_loader(dataset, args.batch_size, args.ddp, shuffle=shuffle,
                         num_workers=args.num_workers, drop_last=drop_last)
    return loader, dataset


def attribute_report(args, specs, split="train"):
    """Count how many images match each spec (and their pairwise intersections).

    Used by scripts/inspect_attributes.py to pick feature pairs that are common
    enough to train the A-and-B baseline on, which the proposal flags as the
    main data-scarcity risk.
    """
    base = _load_celeba(args.data_root, split, args.image_size)
    total = len(base)
    report = {"total": total, "counts": {}, "intersections": {}}
    index_sets = {}
    for spec in specs:
        idx = set(get_attr_indices(base, spec))
        index_sets[spec] = idx
        report["counts"][spec] = len(idx)
    specs = list(specs)
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            report["intersections"][f"{a} & {b}"] = len(index_sets[a] & index_sets[b])
    return report