"""Copy files into an exp_results folder so the orchestrator pulls them.
Usage: python copy_results.py <src>... <dest_dir>"""
import os
import shutil
import sys

dest = sys.argv[-1]
os.makedirs(dest, exist_ok=True)
for src in sys.argv[1:-1]:
    shutil.copy(src, dest)
    print("copied", src, "->", dest)
