import subprocess
import time
import sys
import shutil
import torch
from datetime import timedelta
from pathlib import Path
from models.config import SEEDS, ARCHITECTURES, MODEL_CONFIGS, TRAINING_CONFIGS

TORCHRUN_EXECUTABLE = shutil.which("torchrun") or "torchrun"
PYTHON_EXECUTABLE = sys.executable

# Detect Hardware
GPU_COUNT = torch.cuda.device_count()
IS_CPU_ONLY = (GPU_COUNT == 0)

USE_DDP = GPU_COUNT > 1

if IS_CPU_ONLY:
	N_PROC = 1
	DEVICE_NAME = "CPU"
	RUNNER_MODE = "Standard Python (CPU)"
elif USE_DDP:
	N_PROC = GPU_COUNT
	DEVICE_NAME = f"{GPU_COUNT}x {torch.cuda.get_device_name(0)}"
	RUNNER_MODE = f"PyTorch DDP via torchrun (nproc={N_PROC})"
else:
	N_PROC = 1
	DEVICE_NAME = f"1x {torch.cuda.get_device_name(0)}"
	RUNNER_MODE = "Standard Python (Single GPU)"

print(f"Hardware detected: {DEVICE_NAME}")
print(f"-> Mode: {RUNNER_MODE}")

ACTIVE_SIZES = ["tiny", "small", "base"]
DATA_DIRS = ["./data/processed_ag_news", "./data/processed_fineweb"]


def prepare_data():
	for dir_path in DATA_DIRS:
		path = Path(dir_path)

		if path.is_dir() and any(path.iterdir()):
			print(f"Data dir {path} already exists. Continuing...")
			continue

		print(f"Fetching data dir {path} ...")
		dataset_name = path.name.replace("processed_", "")

		cmd = [
			PYTHON_EXECUTABLE,
			"experiments/prepare_datasets.py",
			"--task", dataset_name,
			"--tokens", "30000000"
		]

		try:
			subprocess.run(cmd, check=True)
			print(f"Fetched {dataset_name} successfully")
		except subprocess.CalledProcessError as e:
			print(f"Error during fetch of dataset {dataset_name}: {e}")


def run_benchmark():
	print(f"Starting benchmark run...")
	print(
		f"RUNS: {len(TRAINING_CONFIGS)} Tasks x {len(ARCHITECTURES)} Architectures x {len(ACTIVE_SIZES)} Sizes x {len(SEEDS)} Seeds")
	print("-" * 60)

	total_start = time.time()
	success_count = 0
	fail_count = 0

	for task in TRAINING_CONFIGS.keys():
		for size in ACTIVE_SIZES:
			for seed in SEEDS:
				for arch in ARCHITECTURES:
					print(f"\n{'=' * 50}")
					print(f"TASK: {task}")
					print(f"RUN: Arch={arch} | Size={size} | Seed={seed}")
					print(f"MODE: {RUNNER_MODE}")
					print(f"{'=' * 50}")

					start_run = time.time()

					cmd = []

					if USE_DDP:
						# Construct torchrun command for Multi-GPU
						cmd = [
							TORCHRUN_EXECUTABLE,
							"--nproc_per_node", str(N_PROC),
							"runner.py"
						]
					else:
						# Construct standard python command for CPU or Single GPU
						cmd = [
							PYTHON_EXECUTABLE,
							"runner.py"
						]

					# Append arguments that apply to both modes
					cmd.extend([
						"--task", task,
						"--arch", arch,
						"--size", size,
						"--seed", str(seed),
						"--compile"
					])

					if IS_CPU_ONLY:
						if "--compile" in cmd:
							cmd.remove("--compile")

					try:
						subprocess.run(cmd, check=True)
						duration = time.time() - start_run
						print(f"FINISHED in {duration:.1f}s")
						success_count += 1

					except subprocess.CalledProcessError:
						print(f"ERROR in {arch} (Seed {seed})!")
						fail_count += 1
						continue
					except KeyboardInterrupt:
						print("\nEXITED manually")
						sys.exit(1)

	total_time = time.time() - total_start
	print("-" * 60)
	print(f"Benchmark completed!")
	print(f"Total duration: {str(timedelta(seconds=int(total_time)))}")
	print(f"SUCCESS: {success_count} | ERROR: {fail_count}")


if __name__ == "__main__":
	prepare_data()
	run_benchmark()