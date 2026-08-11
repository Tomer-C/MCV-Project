readme_content = """# Distributed GPU Cluster Launcher

Minimalist tool to run PyTorch workloads across multiple Azure VMs.

## Setup
1. `pip install python-dotenv`
2. Create `cluster_config.env` with your SSH details (IPs, keys).

## Usage
Syncs local code to all available servers and runs them concurrently:

```bash
python run_cluster.py --gpus --master-idx --args 
