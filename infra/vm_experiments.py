"""Run the proposal's experiments end-to-end on the training machine.

This file lives on and runs ON the GPU VM (pushed there with the rest of the
code by infra/vm_orchestrate.py, but it is plain Python -- you can also run it
yourself from the project root). It executes, fastest first:

  no training needed
    ckpt_metrics     per-checkpoint recon/transfer curves for every run
    negatives        each IGN on has/lacks test images, projector + final stage
    pair_compose     checkpoint-stage composition matrices + joint-rate
                     heatmaps + projector/sharpener schedules, per feature pair
    alt_pairs        full alternating-projection experiment (phases 2+4 sans
                     baseline) for the chosen pairs, noise- and real-init
    alt_stage_sweep  the main pair alternated at several training stages
    alt_k3           three-model alternation, all K! orders
    damped_sweep     damped iteration x <- (1-a)x + a f(x), several alphas

  training required (run last; skipped with --skip_training)
    train_baseline   f_{A and B} trained on the intersection subset
    compare_baseline phase 4: alternating vs the trained baseline (with FID)
    tightness_polish resume Male from the projector stage with a heavy
                     tightness weight, then re-measure feature transfer

Everything is written under exp_results/: images, json, logs -- no weights.
exp_results/status.json is updated after every step so a remote monitor can
poll progress cheaply.

    python -m infra.vm_experiments                 # everything
    python -m infra.vm_experiments --skip_training
    python -m infra.vm_experiments --only negatives,alt_k3
    python -m infra.vm_experiments --list
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------- config

RUNS = {  # feature -> run directory under runs/
    "Male": "Male_500_withsamples",
    "Smiling": "Smiling_paper500",
    "Blond_Hair": "Blond_Hair_500",
    "Young": "Young_500",
}
PAIRS = [("Male", "Smiling"), ("Male", "Blond_Hair"), ("Smiling", "Blond_Hair"),
         ("Male", "Young"), ("Smiling", "Young"), ("Blond_Hair", "Young")]
MAIN_PAIR = ("Male", "Smiling")          # gets the baseline + FID treatment
ALT_PAIRS = PAIRS[:3]                    # pairs given the full run_experiment
K3 = ("Male", "Smiling", "Blond_Hair")
PROJECTOR_EPOCH = 75                     # transfer peaks early (see TRAINING.md)
STAGE_SWEEP = [25, 75, 150, 10 ** 9]     # 1e9 -> resolves to the last ckpt

OUT = "exp_results"
PY = sys.executable


# ---------------------------------------------------------------- helpers

def proj_dir():
    """This file sits in infra/ under the project root; run everything from the root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def as_module(script):
    """'experiments/per_network/x.py' -> 'experiments.per_network.x' (for python -m)."""
    if script.endswith(".py"):
        script = script[:-3].replace("\\", "/").replace("/", ".")
    return script


def run_dir(feature):
    return os.path.join("runs", RUNS[feature])


def spec_of(feature):
    """The attr_spec the run was actually trained on (handles ~Young etc.)."""
    with open(os.path.join(run_dir(feature), "args.json")) as fh:
        return json.load(fh)["attr_spec"]


def ckpt_at(feature, epoch):
    """Path of the numbered checkpoint closest to `epoch`."""
    cands = []
    for p in glob.glob(os.path.join(run_dir(feature), "ckpt_[0-9]*.pt")):
        m = re.search(r"ckpt_(\d+)\.pt$", p)
        if m:
            cands.append((abs(int(m.group(1)) - epoch), int(m.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"no numbered checkpoints in {run_dir(feature)}")
    cands.sort()
    return cands[0][2], cands[0][1]


STATUS = {"overall": "running", "started": time.strftime("%F %T"), "experiments": []}
STATUS_FILE = "status.json"  # overridable so ad-hoc batches don't clobber the main run


def set_status(name, state, note=""):
    for e in STATUS["experiments"]:
        if e["name"] == name:
            e.update(state=state, note=note, at=time.strftime("%F %T"))
            break
    else:
        STATUS["experiments"].append(
            {"name": name, "state": state, "note": note, "at": time.strftime("%F %T")})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, STATUS_FILE), "w") as fh:
        json.dump(STATUS, fh, indent=2)


def sh(name, args, cwd=None, env_extra=None):
    """Run a subordinate script, teeing output to the experiment's log file."""
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    log = os.path.join(OUT, name, "log.txt")
    env = dict(os.environ)
    env["PYTHONPATH"] = proj_dir()
    if env_extra:
        env.update(env_extra)
    # scripts are package modules run with -m; PYTHONPATH makes them importable
    # even when cwd is an output subdirectory
    module = as_module(args[0])
    with open(log, "a") as fh:
        fh.write(f"\n$ python -m {module} {' '.join(args[1:])}\n")
        fh.flush()
        r = subprocess.run([PY, "-m", module] + args[1:], cwd=cwd or proj_dir(), env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} exited {r.returncode}; see {log}")


def classifier_path():
    return os.path.abspath(os.path.join("runs", "classifier", "classifier.pt"))


# ------------------------------------------------------------- experiments

def exp_ckpt_metrics():
    """Per-checkpoint test metrics for every run; copies curves into OUT."""
    import shutil
    for feat, rd in RUNS.items():
        sh("ckpt_metrics", ["experiments/per_network/rebuild_loss_curves.py", "--run_dir", run_dir(feat),
                            "--classifier", classifier_path()])
        for f in ("loss_curves.png", "ckpt_metrics.json"):
            src = os.path.join(run_dir(feat), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(OUT, "ckpt_metrics", f"{feat}_{f}"))


def exp_negatives():
    """Each IGN on with/without-feature test images, two stages."""
    for feat in RUNS:
        for label, epoch in (("proj", PROJECTOR_EPOCH), ("final", 10 ** 9)):
            ck, e = ckpt_at(feat, epoch)
            d = os.path.join(OUT, "negatives", f"{feat}_e{e:04d}_{label}")
            os.makedirs(d, exist_ok=True)
            sh("negatives", ["experiments/per_network/apply_to_negatives.py", "--feature", feat.split(",")[0],
                             "--ckpt", os.path.abspath(ck),
                             "--classifier", classifier_path(),
                             "--data_root", os.path.abspath("data")],
               cwd=d)


def exp_pair_compose():
    """Stage-combination matrices, heatmaps and schedules for every pair."""
    for a, b in PAIRS:
        out = os.path.join(OUT, "pair_compose", f"{a}__{b}")
        sh("pair_compose", ["experiments/per_network/cross_checkpoint_experiments.py",
                            "--run_a", run_dir(a), "--run_b", run_dir(b),
                            "--max_epochs", "4", "--classifier", classifier_path(),
                            "--out_dir", out])


def _run_experiment(name, feats, epochs, tag, extra):
    models, names = [], []
    for f, e in zip(feats, epochs):
        ck, _ = ckpt_at(f, e)
        models.append(ck)
        names.append(f"f_{f}")
    spec = ",".join(spec_of(f) for f in feats)
    sh(name, ["experiments/method1_single_ign/run_experiment.py", "--models", *models, "--names", *names,
              "--intersection_spec", spec, "--classifier", classifier_path(),
              "--out_dir", os.path.join(OUT, name), "--tag", tag] + extra)


def exp_alt_pairs():
    """Alternating projection + order sensitivity for the chosen pairs."""
    for a, b in ALT_PAIRS:
        _run_experiment("alt_pairs", (a, b), [PROJECTOR_EPOCH] * 2,
                        f"{a}__{b}_noise",
                        ["--num_samples", "256", "--num_iters", "15", "--skip_fid"])
    a, b = MAIN_PAIR  # real-image initialisation for the main pair
    _run_experiment("alt_pairs", (a, b), [PROJECTOR_EPOCH] * 2,
                    f"{a}__{b}_realinit",
                    ["--num_samples", "256", "--num_iters", "15", "--skip_fid",
                     "--init", "real"])


def exp_alt_stage_sweep():
    """The main pair alternated with both models at several training stages."""
    a, b = MAIN_PAIR
    for e in STAGE_SWEEP:
        _, ea = ckpt_at(a, e)
        _run_experiment("alt_stage_sweep", (a, b), [e, e], f"stage_e{ea:04d}",
                        ["--num_samples", "256", "--num_iters", "15",
                         "--skip_fid", "--skip_orders"])


def exp_alt_k3():
    _run_experiment("alt_k3", K3, [PROJECTOR_EPOCH] * 3, "Male_Smiling_Blond",
                    ["--num_samples", "256", "--num_iters", "15", "--skip_fid"])


def exp_damped_sweep():
    """Joint-residual curves of the damped iteration for several alphas."""
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ign.alternating import damped_alternating_projection
    from ign.models import load_ign
    from ign.utils import sample_freq_noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = MAIN_PAIR
    fa, _ = load_ign(ckpt_at(a, PROJECTOR_EPOCH)[0], device)
    fb, _ = load_ign(ckpt_at(b, PROJECTOR_EPOCH)[0], device)
    import torch as _t
    ck = _t.load(ckpt_at(a, PROJECTOR_EPOCH)[0], map_location=device, weights_only=False)
    if "freq_stats" in ck:
        stats = {k: v.to(device) for k, v in ck["freq_stats"].items()}
        x0 = sample_freq_noise(stats, 128)
    else:
        x0 = torch.randn(128, 3, 64, 64, device=device)

    d = os.path.join(OUT, "damped_sweep")
    os.makedirs(d, exist_ok=True)
    curves = {}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        r = damped_alternating_projection([fa, fb], x0, num_iters=15, alpha=alpha)
        curves[alpha] = r["metrics"]["joint_residual"].mean(1).tolist()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha, ys in curves.items():
        ax.plot(range(1, len(ys) + 1), ys, marker=".", label=f"alpha={alpha}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean joint fixed-point residual")
    ax.set_title(f"damped alternation, {a}+{b} @ e{PROJECTOR_EPOCH}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(d, "damping.png"), dpi=120)
    with open(os.path.join(d, "damping.json"), "w") as fh:
        json.dump(curves, fh, indent=2)


def exp_train_baseline(args):
    """Phase 3: train f_{A and B} on the intersection subset."""
    a, b = MAIN_PAIR
    spec = ",".join(spec_of(f) for f in (a, b))
    sh("train_baseline", ["train/train_ign.py", f"--attr_spec={spec}",
                          "--run_name", "Baseline_" + "_AND_".join(MAIN_PAIR),
                          "--num_epochs", str(args.baseline_epochs),
                          "--save_every", "25"])


def exp_compare_baseline(args):
    """Phase 4: alternating vs the trained baseline, on the same x0."""
    a, b = MAIN_PAIR
    base = os.path.join("runs", "Baseline_" + "_AND_".join(MAIN_PAIR), "ckpt_last.pt")
    for tag, extra in (("noise", []), ("realinit", ["--init", "real"])):
        models = [ckpt_at(a, PROJECTOR_EPOCH)[0], ckpt_at(b, PROJECTOR_EPOCH)[0]]
        sh("compare_baseline",
           ["experiments/method1_single_ign/run_experiment.py", "--models", *models, "--names", f"f_{a}", f"f_{b}",
            "--baseline", base,
            "--intersection_spec", ",".join(spec_of(f) for f in (a, b)),
            "--classifier", classifier_path(),
            "--num_samples", "1024", "--num_iters", "15",
            "--out_dir", os.path.join(OUT, "compare_baseline"), "--tag", tag] + extra)


def exp_tightness_polish(args):
    """Resume Male from the projector stage with a heavy tightness weight,
    then re-measure feature transfer at the polished checkpoint."""
    feat = "Male"
    ck, e = ckpt_at(feat, PROJECTOR_EPOCH)
    sh("tightness_polish", ["train/train_ign.py", f"--attr_spec={spec_of(feat)}",
                            "--run_name", "Male_polish_d10",
                            "--resume", ck,
                            "--num_epochs", str(e + args.polish_epochs),
                            "--discriminative_loss_factor", "10",
                            "--save_every", "10"])
    d = os.path.join(OUT, "tightness_polish", "negatives_after")
    os.makedirs(d, exist_ok=True)
    sh("tightness_polish", ["experiments/per_network/apply_to_negatives.py", "--feature", feat,
                            "--ckpt", os.path.abspath(
                                os.path.join("runs", "Male_polish_d10", "ckpt_last.pt")),
                            "--classifier", classifier_path(),
                            "--data_root", os.path.abspath("data")],
       cwd=d)


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser("run all proposal experiments on the VM")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--baseline_epochs", type=int, default=120)
    ap.add_argument("--polish_epochs", type=int, default=40)
    ap.add_argument("--status_file", default="status.json",
                    help="status filename under exp_results/ (ad-hoc batches "
                         "should use their own, e.g. status_adaptive1.json)")
    ap.add_argument("--spec", default="",
                    help="JSON file with an ad-hoc experiment batch: a list of "
                         '{"name": ..., "args": [script, arg, ...], "cwd": optional '
                         "subdir under exp_results/} entries, run in order; script "
                         "is a module (train.train_ign) or a path from the project "
                         "root (train/train_ign.py)")
    args = ap.parse_args()

    global STATUS_FILE
    STATUS_FILE = args.status_file

    os.chdir(proj_dir())
    if args.spec:
        with open(args.spec) as fh:
            batch = json.load(fh)
        for e in batch:
            set_status(e["name"], "pending")
        for e in batch:
            set_status(e["name"], "running")
            t0 = time.time()
            try:
                cwd = None
                if e.get("cwd"):
                    cwd = os.path.join(proj_dir(), OUT, e["cwd"])
                    os.makedirs(cwd, exist_ok=True)
                sh(e["name"], e["args"], cwd=cwd)
                set_status(e["name"], "done", f"{time.time() - t0:.0f}s")
            except Exception as exc:
                set_status(e["name"], "failed", str(exc)[:300])
        STATUS["overall"] = "done"
        set_status("_run", "done")
        print("BATCH DONE")
        return
    experiments = [
        ("ckpt_metrics", lambda: exp_ckpt_metrics()),
        ("negatives", lambda: exp_negatives()),
        ("pair_compose", lambda: exp_pair_compose()),
        ("alt_pairs", lambda: exp_alt_pairs()),
        ("alt_stage_sweep", lambda: exp_alt_stage_sweep()),
        ("alt_k3", lambda: exp_alt_k3()),
        ("damped_sweep", lambda: exp_damped_sweep()),
        ("train_baseline", lambda: exp_train_baseline(args)),
        ("compare_baseline", lambda: exp_compare_baseline(args)),
        ("tightness_polish", lambda: exp_tightness_polish(args)),
    ]
    training = {"train_baseline", "compare_baseline", "tightness_polish"}
    if args.list:
        for n, _ in experiments:
            print(n + ("  [training]" if n in training else ""))
        return
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    for name, fn in experiments:
        if only and name not in only:
            continue
        if args.skip_training and name in training:
            set_status(name, "skipped", "--skip_training")
            continue
        set_status(name, "running")
        t0 = time.time()
        try:
            fn()
            set_status(name, "done", f"{time.time() - t0:.0f}s")
        except Exception as e:
            set_status(name, "failed", str(e)[:300])
        print(f"[{time.strftime('%T')}] {name}: {STATUS['experiments'][-1]['state']}"
              if STATUS["experiments"] else name, flush=True)

    STATUS["overall"] = "done"
    STATUS["finished"] = time.strftime("%F %T")
    set_status("_run", "done")
    print("ALL DONE")


if __name__ == "__main__":
    main()
