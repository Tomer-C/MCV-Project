import os
import argparse
import subprocess
from dotenv import load_dotenv

def check_server_up(ip, user, key):
    # Attempt a quick SSH connection to verify the server is online and accessible
    cmd = [
        "ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=no", f"{user}@{ip}", "echo 1"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False

def main():
    # Setup command-line arguments for the script
    parser = argparse.ArgumentParser(description="Distributed GPU runner")
    parser.add_argument("filename", type=str, help="Python script to run")
    parser.add_argument("--args", type=str, default="", help="Arguments for the script")
    parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs/nodes to use")
    parser.add_argument("--master-idx", type=int, default=0, help="Index of master node")
    parser.add_argument("--env", type=str, default="cluster_config.env", help="Path to env configuration")

    parsed = parser.parse_args()
    
    # Load server configurations from the specified .env file
    load_dotenv(parsed.env)
    project_path = os.getenv("PROJECT_PATH", ".")

    available_nodes = []
    idx = 0
    
    # Iterate through all configured nodes in the .env file to check which ones are currently online
    while True:
        ip = os.getenv(f"NODE_{idx}_IP")
        if not ip: # Stop searching when we run out of defined nodes
            break
        user = os.getenv(f"NODE_{idx}_USER")
        key = os.path.expanduser(os.getenv(f"NODE_{idx}_KEY", "~/.ssh/id_rsa"))
        
        if check_server_up(ip, user, key):
            available_nodes.append({"idx": idx, "ip": ip, "user": user, "key": key})
        idx += 1

    # If no servers are reachable, offer to run the script locally instead
    if not available_nodes:
        choice = input("No servers are available. Run locally? (y/n): ")
        if choice.lower() == 'y':
            subprocess.run(f"python {parsed.filename} {parsed.args}", shell=True)
        return

    # If fewer servers are online than requested, cap the execution to the available amount
    if len(available_nodes) < parsed.gpus:
        print(f"Notice: Only {len(available_nodes)} of {parsed.gpus} requested servers are available. Using maximum available.")
        target_nodes = available_nodes
    else:
        target_nodes = available_nodes[:parsed.gpus]

    # Assign the master node. If the requested master is down, default to the first available node.
    master_node = next((n for n in target_nodes if n["idx"] == parsed.master_idx), target_nodes[0])
    master_ip = master_node["ip"]
    master_port = os.getenv("MASTER_PORT", "29500")
    actual_gpus = len(target_nodes)

    processes = []

    # Launch the process on each available node
    for logical_rank, node in enumerate(target_nodes):
        ip = node["ip"]
        user = node["user"]
        key = node["key"]

        # 1. Sync the latest code from the local machine to the remote worker
        rsync_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=no",
            "--exclude", ".git", "--exclude", "__pycache__",
            f"{project_path}/", f"{user}@{ip}:{project_path}/"
        ]
        subprocess.run(rsync_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Construct the PyTorch distributed command with proper networking parameters
        torchrun_cmd = (
            f"cd {project_path} && "
            f"torchrun --nproc_per_node=1 "
            f"--nnodes={actual_gpus} "
            f"--node_rank={logical_rank} "
            f"--master_addr={master_ip} "
            f"--master_port={master_port} "
            f"{parsed.filename} {parsed.args}"
        )

        # 3. Wrap the torchrun command in SSH to execute it on the remote machine
        ssh_cmd = [
            "ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
            f"{user}@{ip}", torchrun_cmd
        ]

        # Start the remote execution asynchronously
        p = subprocess.Popen(ssh_cmd)
        processes.append((logical_rank, ip, p))

    # Wait for all remote processes to finish their training/execution
    for rank, ip, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Node {rank} ({ip}) failed with exit code {p.returncode}")

if __name__ == "__main__":
    main()
