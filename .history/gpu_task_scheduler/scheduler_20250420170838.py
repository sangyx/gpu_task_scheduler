import subprocess
import time
import os
from collections import defaultdict

class GPUTaskScheduler:
    def __init__(self, wait_interval=10, allowed_gpu_ids=None, max_tasks_per_gpu=1):
        self.wait_interval = wait_interval
        self.allowed_gpu_ids = allowed_gpu_ids if allowed_gpu_ids is not None else []
        self.max_tasks_per_gpu = max_tasks_per_gpu
        self.gpu_task_counts = defaultdict(int)

    def execute_task_on_gpu(self, gpu_id, command):
        print(f"Executing task on GPU {gpu_id}: {command}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.gpu_task_counts[gpu_id] += 1
        try:
            subprocess.run(command, shell=True, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Task failed with error: {e}")
        finally:
            self.gpu_task_counts[gpu_id] -= 1

    def get_available_gpu(self):
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True
            ).strip().splitlines()
            busy_gpus = set(result)

            uuid_map = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True
            )
            for line in uuid_map.strip().splitlines():
                idx, uuid = line.split(',')
                idx = int(idx.strip())
                uuid = uuid.strip()
                if (not self.allowed_gpu_ids or idx in self.allowed_gpu_ids) and \
                   (uuid not in busy_gpus or self.gpu_task_counts[idx] < self.max_tasks_per_gpu):
                    return idx
        except subprocess.CalledProcessError as e:
            print(f"Error checking GPU status: {e}")
        return None

    def run_tasks(self, tasks):
        for command in tasks:
            while True:
                gpu_id = self.get_available_gpu()
                if gpu_id is not None:
                    print(f"Found available GPU: {gpu_id}")
                    self.execute_task_on_gpu(gpu_id, command)
                    break
                else:
                    print("No available GPU or max task limit reached. Waiting...")
                    time.sleep(self.wait_interval)