"""Train the CelebA attribute classifier used to evaluate generated images.

This is the measuring device for the central claim. When the alternating
scheme outputs x*, we need an independent judgement of whether x* actually
has attribute A and attribute B. The classifier is trained on the FULL
(unfiltered) CelebA train split so it is not biased toward either subset, and
validated on the official val split.

  python train_classifier.py --epochs 8
"""

import argparse
import os

import torch
import torch.nn as nn

from ign_data import CELEBA_ATTRS, get_data_loader
from ign_models import AttributeClassifier
from ign_utils import resolve_device, set_seed


def get_args_parser():
    p = argparse.ArgumentParser("Train CelebA attribute classifier")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--epochs", default=8, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--out", default="runs/classifier/classifier.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--ddp", action="store_true", help="unused; kept for loader signature")
    p.add_argument("--max_steps_per_epoch", default=0, type=int)
    return p


def main(args):
    set_seed(args.seed)
    device = resolve_device(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train_loader, train_ds = get_data_loader(args, "all", split="train")
    val_loader, val_ds = get_data_loader(args, "all", split="valid",
                                         shuffle=False, drop_last=False)
    print(f"classifier: train={len(train_ds)} val={len(val_ds)} device={device}")

    model = AttributeClassifier(num_attrs=len(CELEBA_ATTRS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  e{epoch} s{step} loss={loss.item():.4f}")

        acc, per_attr = evaluate(model, val_loader, device, args.max_steps_per_epoch)
        print(f"epoch {epoch}: train_loss={running/max(step+1,1):.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(),
                        "val_acc": acc,
                        "per_attr_acc": per_attr,
                        "attrs": CELEBA_ATTRS,
                        "epoch": epoch}, args.out)
            print(f"  saved -> {args.out}")

    print(f"best val accuracy: {best_acc:.4f}")
    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    print("\nper-attribute val accuracy (relevant ones):")
    for name in ["Smiling", "Eyeglasses", "Male", "Blond_Hair", "Black_Hair",
                 "Wearing_Hat", "Young", "Heavy_Makeup", "Mustache", "No_Beard"]:
        i = CELEBA_ATTRS.index(name)
        print(f"  {name:20s} {ckpt['per_attr_acc'][i]:.4f}")


@torch.no_grad()
def evaluate(model, loader, device, max_steps=0):
    model.eval()
    correct = torch.zeros(len(CELEBA_ATTRS), device=device)
    total = 0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        pred = (model(x) > 0).float()
        correct += (pred == y).float().sum(0)
        total += x.shape[0]
    per_attr = (correct / max(total, 1)).cpu().tolist()
    return float(sum(per_attr) / len(per_attr)), per_attr


if __name__ == "__main__":
    main(get_args_parser().parse_args())
