"""Integration test: runs train/train_ign.py and experiments/method1_single_ign/run_experiment.py for real, on a
synthetic stand-in for CelebA. Catches wiring bugs that unit tests miss
(argument plumbing, checkpoint round-trips, file outputs, figure generation).

Uses a monkeypatched dataset so it needs no download and finishes in ~1 minute
on CPU. It proves the pipeline executes; it says nothing about image quality.

  python scripts/integration_test.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# A fake dataset backed by random tensors. Injected into the child processes as
# `usercustomize`, which CPython imports automatically at startup when it is
# importable from PYTHONPATH (unlike .pth files, which only work in site-packages).
FAKE_DATA = '''
import torch, random
from torch.utils.data import Dataset
import ign.data as _d

class _Fake(Dataset):
    def __init__(self, n=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 64, 64, generator=g) * 2 - 1
        self.a = (torch.rand(n, 40, generator=g) > 0.5).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.a[i]

def _fake_filtered(args, spec, split="train"):
    seed = abs(hash((spec, split))) % 10000
    return _Fake(64, seed)

_d.get_filtered_dataset = _fake_filtered

def _fake_loader(args, spec, split="train", shuffle=True, drop_last=True):
    ds = _fake_filtered(args, spec, split)
    return _d.make_loader(ds, args.batch_size, False, shuffle=shuffle,
                          num_workers=0, drop_last=drop_last), ds

_d.get_data_loader = _fake_loader
'''


def run(cmd, cwd, env):
    print(f"\n$ {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print("STDERR:", r.stderr[-4000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-6:]
    print("\n".join("    " + l for l in tail))
    return r.stdout


def main():
    tmp = tempfile.mkdtemp(prefix="ign_integration_")
    print(f"workspace: {tmp}")
    try:
        # `usercustomize` is imported automatically by the site module at startup
        patch_dir = os.path.join(tmp, "patch")
        os.makedirs(patch_dir)
        with open(os.path.join(patch_dir, "usercustomize.py"), "w") as fh:
            fh.write(FAKE_DATA)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([ROOT, patch_dir])
        env.pop("PYTHONNOUSERSITE", None)
        common = ["--data_root", os.path.join(tmp, "data"), "--batch_size", "8",
                  "--num_epochs", "2", "--num_workers", "0", "--device", "cpu",
                  "--max_steps_per_epoch", "3", "--save_every", "1", "--sample_every", "1"]

        for spec, name in [("Smiling", "f_A"), ("Eyeglasses", "f_B"),
                           ("Smiling,Eyeglasses", "f_AB")]:
            run([sys.executable, "-m", "train.train_ign", "--attr_spec", spec, "--run_name", name,
                 "--out_dir", os.path.join(tmp, "runs")] + common, ROOT, env)
            ck = os.path.join(tmp, "runs", name, "ckpt_last.pt")
            assert os.path.exists(ck), f"missing checkpoint {ck}"
        print("\n  [checkpoints written]")

        run([sys.executable, "-m", "train.train_classifier", "--epochs", "1", "--batch_size", "8",
             "--num_workers", "0", "--device", "cpu", "--max_steps_per_epoch", "3",
             "--data_root", os.path.join(tmp, "data"),
             "--out", os.path.join(tmp, "runs", "classifier", "classifier.pt")], ROOT, env)

        run([sys.executable, "-m", "experiments.method1_single_ign.run_experiment",
             "--models", os.path.join(tmp, "runs", "f_A", "ckpt_last.pt"),
             os.path.join(tmp, "runs", "f_B", "ckpt_last.pt"),
             "--names", "f_A", "f_B",
             "--baseline", os.path.join(tmp, "runs", "f_AB", "ckpt_last.pt"),
             "--intersection_spec", "Smiling,Eyeglasses",
             "--classifier", os.path.join(tmp, "runs", "classifier", "classifier.pt"),
             "--num_iters", "4", "--num_samples", "16", "--skip_fid",
             "--data_root", os.path.join(tmp, "data"), "--device", "cpu",
             "--out_dir", os.path.join(tmp, "results"), "--tag", "test"], ROOT, env)

        rj = os.path.join(tmp, "results", "test", "results.json")
        with open(rj) as fh:
            res = json.load(fh)
        for key in ["alternating", "attr_rates", "orders", "paired_vs_baseline",
                    "order_disagreement", "damped"]:
            assert key in res, f"results.json missing {key}"
        assert len(res["alternating"]["step_norm_mean"]) == 4
        assert "f_B o f_A" in res["orders"] and "f_A o f_B" in res["orders"]
        print(f"\n  [results.json complete: {list(res)}]")

        run([sys.executable, "-m", "experiments.method1_single_ign.make_figures",
             "--results", os.path.join(tmp, "results", "test")], ROOT, env)
        figs = os.listdir(os.path.join(tmp, "results", "test", "figures"))
        for expected in ["convergence.png", "attr_rates.png", "orders.png",
                         "trajectory.png", "damping.png"]:
            assert expected in figs, f"missing figure {expected} (have {figs})"
        print(f"\n  [figures: {sorted(figs)}]")

        print("\nIntegration test PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
