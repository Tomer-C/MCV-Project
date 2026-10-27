"""Run a train_commuting_pair.py hyperparameter sweep, evaluating as it goes.

Reads a JSON spec, trains each run in order (one at a time -- a single run
already saturates the GPU), and after each finishes runs
commuting_sweep_eval.py so results accumulate while the rest of the sweep
trains. Runs whose final snapshot already exists are skipped, so an
interrupted sweep resumes by re-launching the same command.

Spec format
-----------
  {
    "ckpt_a": "runs/Male_500_withsamples/ckpt_0075.pt",
    "ckpt_b": "runs/Smiling_paper500/ckpt_0075.pt",
    "baseline": "runs/Baseline_Male_AND_Smiling/ckpt_last.pt",   (optional)
    "intersection_spec": "Male,Smiling",
    "eval_out": "exp_results/commuting_sweep/male_smiling",
    "extra_eval_runs": ["runs/Male_Smiling_commuting_probe"],     (optional)
    "common_args": ["--num_epochs", "50", "--save_every", "5"],
    "runs": [{"name": "ms_l1_lam0_lr1e-4", "args": ["--lambda_commute", "0"]}, ...]
  }

  python commuting_sweep.py --spec sweep_male_smiling.json
  (on the VM: python vm_orchestrate.py exec commuting_sweep.py -- --spec <spec>)
"""

import argparse
import json
import os
import subprocess
import sys
import time


def arg_value(args, flag, default=None):
    """Value of the last occurrence of `flag` -- argparse's last-wins rule, so
    a run's own args can override common_args."""
    idx = [i for i, a in enumerate(args) if a == flag]
    return args[idx[-1] + 1] if idx else default


def main():
    p = argparse.ArgumentParser("commuting-loss sweep runner")
    p.add_argument("--spec", required=True)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--only", nargs="*", default=None, help="run names to include")
    a = p.parse_args()

    spec = json.load(open(a.spec))
    log_dir = os.path.join(spec["eval_out"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(spec["eval_out"], "status.json")
    status = {"spec": a.spec, "started": time.strftime("%F %T"), "runs": {}}

    def set_status(name, state, **extra):
        status["runs"][name] = {"state": state, "time": time.strftime("%F %T"), **extra}
        with open(status_path, "w") as fh:
            json.dump(status, fh, indent=1)

    def evaluate(run_dirs):
        cmd = [sys.executable, "commuting_sweep_eval.py",
               "--naive_a", spec["ckpt_a"], "--naive_b", spec["ckpt_b"],
               "--intersection_spec", spec["intersection_spec"],
               "--out", spec["eval_out"], "--runs", *run_dirs]
        if spec.get("baseline"):
            cmd += ["--baseline", spec["baseline"]]
        with open(os.path.join(log_dir, "eval.log"), "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)

    runs = [r for r in spec["runs"] if a.only is None or r["name"] in a.only]
    for r in runs:
        set_status(r["name"], "queued")
    if spec.get("extra_eval_runs"):
        evaluate(spec["extra_eval_runs"])

    for r in runs:
        name = r["name"]
        args = spec.get("common_args", []) + r.get("args", [])
        n_epochs = int(arg_value(args, "--num_epochs", 30))
        run_dir = os.path.join(a.out_dir, name)
        final = os.path.join(run_dir, f"ckpt_B_e{n_epochs:04d}.pt")
        if os.path.exists(final):
            print(f"[sweep] {name}: already trained, skipping to eval", flush=True)
        else:
            cmd = [sys.executable, "train_commuting_pair.py",
                   "--ckpt_a", spec["ckpt_a"], "--ckpt_b", spec["ckpt_b"],
                   "--run_name", name, "--out_dir", a.out_dir, *args]
            set_status(name, "training", cmd=" ".join(cmd))
            print(f"[sweep] {name}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            with open(os.path.join(log_dir, f"{name}.log"), "w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                set_status(name, "failed", rc=rc)
                print(f"[sweep] {name}: FAILED rc={rc}", flush=True)
                continue
            print(f"[sweep] {name}: trained in {(time.time() - t0) / 60:.0f} min", flush=True)
        set_status(name, "evaluating")
        rc = evaluate([run_dir])
        set_status(name, "done" if rc == 0 else "eval_failed", rc=rc)

    status["finished"] = time.strftime("%F %T")
    with open(status_path, "w") as fh:
        json.dump(status, fh, indent=1)
    print("[sweep] all done", flush=True)


if __name__ == "__main__":
    main()
