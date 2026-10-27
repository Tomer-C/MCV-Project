Minimalist tool to run PyTorch workloads across multiple Azure VMs.

## Setup
1. `pip install python-dotenv`
2. Fill in `infra/cluster/cluster_config.env` with your SSH details (IPs, keys).

## Usage
Syncs local code to all available servers and runs them concurrently. From the
project root, name the script as a module or as a path from the root:

```bash
python infra/cluster/run_cluster.py train.train_ign --gpus 4 --args "--attr_spec Smiling --ddp"
```
