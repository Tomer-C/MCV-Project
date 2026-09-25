"""Drive vm_experiments.py on the GPU VM from this machine, over SSH.

All VM interaction lives here (paramiko); no manual shelling required.
Configuration is read from vm_config.json next to this file (gitignored):

    {
      "host": "<VM_IP_OR_DNS>",
      "user": "<VM_USER>",
      "password": "<VM_PASSWORD>",
      "remote_root": "/home/<VM_USER>/project",
      "project_dir": "<PROJECT_DIR_NAME>",
      "local_results": "vm_results",
      "remote_python": ""          <- auto-detected and written back
    }

vm_config.json is gitignored on purpose: credentials never enter the repo.
Copy vm_config.template.json to vm_config.json and fill it in.

Commands:
    python vm_orchestrate.py push      # upload current code + vm_experiments.py
    python vm_orchestrate.py start [-- extra vm_experiments args]
    python vm_orchestrate.py status    # status.json + tail of the run log
    python vm_orchestrate.py pull      # download images/reports (never *.pt)
    python vm_orchestrate.py watch     # poll status + pull until done
    python vm_orchestrate.py run       # push + start + watch
    python vm_orchestrate.py stop      # kill the remote run

    # generic runner, for one-off scripts outside vm_experiments.py's fixed
    # suite (e.g. train_commuting_pair.py) -- log at exp_results/<script>.log
    python vm_orchestrate.py exec <script.py> [-- extra script args]
    python vm_orchestrate.py tail <script.py>        # tail its log + pgrep
    python vm_orchestrate.py killscript <script.py>  # pkill -f <script.py>

Only files with these extensions are ever downloaded:
png jpg jpeg json txt csv md log  --  weights stay on the VM.
"""

import json
import os
import posixpath
import stat
import sys
import time

try:
    import paramiko
except ImportError:
    sys.exit("pip install paramiko")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "vm_config.json")
PUSH_FILES = [
    "vm_experiments.py", "train_ign.py", "train_commuting_pair.py",
    "ign_models.py", "ign_utils.py",
    "ign_data.py", "alternating.py", "run_experiment.py", "metrics.py",
    "cross_checkpoint_experiments.py", "apply_to_negatives.py",
    "rebuild_loss_curves.py", "make_feature_grids.py", "make_attr_splits.py",
    "polish_eval.py", "league_eval.py", "diversity_eval.py", "relax_eval.py", "confirm_eval.py", "pairs_relax_eval.py", "copy_results.py", "black_pairs_cfg.json", "triple_eval.py", "real_edit_eval.py", "quad_eval.py", "nosel_eval.py", "reversed_pair_cfg.json", "glasses_pairs_cfg.json",
]
PULL_EXT = {".png", ".jpg", ".jpeg", ".json", ".txt", ".csv", ".md", ".log"}
PY_CANDIDATES = [
    "$HOME/stud_prj/bin/python",
    "$HOME/miniconda3/envs/stud_prj/bin/python",
    "$HOME/anaconda3/envs/stud_prj/bin/python",
    "$HOME/venv/bin/python", "python3",
]


def cfg():
    with open(CONFIG) as fh:
        return json.load(fh)


def connect(c):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(c["host"], username=c["user"], password=c["password"],
                timeout=25, banner_timeout=25, auth_timeout=25)
    return cli


def run(cli, cmd, timeout=60):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace"), \
        out.channel.recv_exit_status()


def remote_proj(c):
    return posixpath.join(c["remote_root"], c["project_dir"])


def detect_python(cli, c):
    if c.get("remote_python"):
        return c["remote_python"]
    for cand in PY_CANDIDATES:
        o, _, rc = run(cli, f'{cand} -c "import torch; print(torch.__version__)"')
        if rc == 0 and o.strip():
            print(f"remote python: {cand} (torch {o.strip()})")
            c["remote_python"] = cand
            with open(CONFIG, "w") as fh:
                json.dump(c, fh, indent=2)
            return cand
    sys.exit("no remote python with torch found; set remote_python in vm_config.json")


def cmd_push(cli, c):
    import glob as _glob
    sftp = cli.open_sftp()
    proj = remote_proj(c)
    batch_specs = [os.path.basename(p)
                   for p in _glob.glob(os.path.join(HERE, "adaptive_*.json"))]
    for f in PUSH_FILES + batch_specs:
        local = os.path.join(HERE, f)
        if not os.path.exists(local):
            print(f"  skip (missing locally): {f}")
            continue
        sftp.put(local, posixpath.join(proj, f))
        print(f"  pushed {f}")
    sftp.close()


def cmd_start(cli, c, extra):
    py = detect_python(cli, c)
    proj = remote_proj(c)
    flags = " ".join(extra)
    cmd = (f"cd {proj} && mkdir -p exp_results && "
           f"nohup setsid {py} vm_experiments.py {flags} "
           f"> exp_results/run.log 2>&1 < /dev/null & echo started_pid=$!")
    try:  # the channel may not close while the detached child lives; don't block
        _, out, _ = cli.exec_command(cmd, timeout=15)
        print(out.readline().strip())
    except Exception:
        pass
    time.sleep(3)
    o, _, _ = run(cli, "pgrep -f vm_experiments.py | head -2 || true")
    print("running as pid(s): " + (o.strip().replace("\n", ", ")
                                   or "NOT RUNNING -- check exp_results/run.log"))


def cmd_exec(cli, c, script, extra):
    """Launch an arbitrary pushed script remotely (nohup, detached), for
    one-off jobs like train_commuting_pair.py that aren't part of
    vm_experiments.py's fixed suite. Log -> exp_results/<script>.log; poll
    with `tail <script>`, stop with `killscript <script>`."""
    py = detect_python(cli, c)
    proj = remote_proj(c)
    flags = " ".join(extra)
    log = f"exp_results/{os.path.splitext(script)[0]}.log"
    cmd = (f"cd {proj} && mkdir -p exp_results && "
           f"nohup setsid {py} {script} {flags} "
           f"> {log} 2>&1 < /dev/null & echo started_pid=$!")
    try:
        _, out, _ = cli.exec_command(cmd, timeout=15)
        print(out.readline().strip())
    except Exception:
        pass
    time.sleep(3)
    o, _, _ = run(cli, f"pgrep -f {script} | head -2 || true")
    print("running as pid(s): " + (o.strip().replace("\n", ", ")
                                   or f"NOT RUNNING -- check {log}"))


def cmd_tail(cli, c, script, n=40):
    proj = remote_proj(c)
    log = f"{proj}/exp_results/{os.path.splitext(script)[0]}.log"
    o, _, _ = run(cli, f"tail -n {n} {log} 2>/dev/null || echo '(no log yet)'")
    print(o)
    o, _, _ = run(cli, f"pgrep -af {script} | grep -v pgrep || true")
    print("process: " + (o.strip() or "not running"))


def cmd_killscript(cli, c, script):
    run(cli, f"pkill -f {script} || true")
    print(f"sent kill to {script}")


def cmd_status(cli, c):
    proj = remote_proj(c)
    o, _, _ = run(cli, f"ls {proj}/exp_results/status*.json 2>/dev/null")
    for sf in o.split():
        raw, _, _ = run(cli, f"cat {sf}")
        if not raw.strip():
            continue
        s = json.loads(raw)
        print(f"[{posixpath.basename(sf)}] overall: {s.get('overall')}  "
              f"(started {s.get('started')})")
        for e in s.get("experiments", []):
            print(f"  {e['name']:22s} {e['state']:8s} {e.get('note', '')}")
    if not o.strip():
        print("no status files yet")
    o, _, _ = run(cli, f"tail -4 {proj}/exp_results/run.log 2>/dev/null")
    if o.strip():
        print("log tail:\n  " + "\n  ".join(o.strip().splitlines()[-4:]))
    o, _, _ = run(cli, "pgrep -af vm_experiments.py | grep -v pgrep || true")
    print("process: " + (o.strip() or "not running"))
    return o.strip()


def cmd_pull(cli, c):
    sftp = cli.open_sftp()
    remote_base = posixpath.join(remote_proj(c), "exp_results")
    local_base = os.path.join(HERE, c.get("local_results", "vm_results"))
    n = 0

    def walk(rdir, ldir):
        nonlocal n
        try:
            entries = sftp.listdir_attr(rdir)
        except FileNotFoundError:
            return
        os.makedirs(ldir, exist_ok=True)
        for ent in entries:
            rp, lp = posixpath.join(rdir, ent.filename), os.path.join(ldir, ent.filename)
            if stat.S_ISDIR(ent.st_mode):
                walk(rp, lp)
            else:
                ext = os.path.splitext(ent.filename)[1].lower()
                if ext not in PULL_EXT or ent.st_size > 50 * 1024 * 1024:
                    continue
                if os.path.exists(lp) and os.path.getsize(lp) == ent.st_size:
                    continue
                sftp.get(rp, lp)
                n += 1

    walk(remote_base, local_base)
    sftp.close()
    print(f"pulled {n} new/changed files -> {local_base}")


def cmd_watch(cli, c, interval=90):
    while True:
        print("=" * 60, time.strftime("%T"))
        proc = cmd_status(cli, c)
        cmd_pull(cli, c)
        o, _, _ = run(cli, f"cat {remote_proj(c)}/exp_results/status.json 2>/dev/null")
        if o.strip() and json.loads(o).get("overall") == "done":
            print("run finished.")
            break
        if not proc and o.strip():
            print("WARNING: process not running but status not 'done' "
                  "(crashed? check run.log)")
            break
        time.sleep(interval)


def cmd_stop(cli, c):
    run(cli, "pkill -f vm_experiments.py || true")
    print("sent kill")


def main():
    args = sys.argv[1:]
    commands = ("push", "start", "status", "pull", "watch", "run", "stop",
               "exec", "tail", "killscript")
    if not args or args[0] not in commands:
        sys.exit(__doc__)
    c = cfg()
    cli = connect(c)
    try:
        if args[0] == "push":
            cmd_push(cli, c)
        elif args[0] == "start":
            cmd_start(cli, c, args[1:])
        elif args[0] == "status":
            cmd_status(cli, c)
        elif args[0] == "pull":
            cmd_pull(cli, c)
        elif args[0] == "watch":
            cmd_watch(cli, c)
        elif args[0] == "stop":
            cmd_stop(cli, c)
        elif args[0] == "run":
            cmd_push(cli, c)
            cmd_start(cli, c, args[1:])
            time.sleep(20)
            cmd_watch(cli, c)
        elif args[0] == "exec":
            # python vm_orchestrate.py exec <script.py> [-- extra script args]
            script, extra = args[1], args[2:]
            if extra and extra[0] == "--":
                extra = extra[1:]
            cmd_exec(cli, c, script, extra)
        elif args[0] == "tail":
            cmd_tail(cli, c, args[1])
        elif args[0] == "killscript":
            cmd_killscript(cli, c, args[1])
    finally:
        cli.close()


if __name__ == "__main__":
    main()
