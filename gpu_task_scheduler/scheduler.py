import subprocess
import time
import os
import threading
from collections import defaultdict


class GPUTaskScheduler:
    """
    A scheduler to manage and dispatch deep learning tasks across available GPUs.

    Parameters:
    - wait_interval (int): Seconds to wait before re-checking GPU availability.
    - allowed_gpu_ids (list[int] or None): Specific GPU IDs allowed for task execution.
    - max_tasks_per_gpu (int): Max concurrent tasks allowed on a single GPU.
    - min_memory (int or float): Minimum free memory (in MiB or as a ratio) required on a GPU to be considered available.
    """

    def __init__(self, wait_interval=30, allowed_gpu_ids=None, max_tasks_per_gpu=1, min_memory=0.8):
        self.wait_interval = wait_interval
        self.allowed_gpu_ids = allowed_gpu_ids if allowed_gpu_ids is not None else []
        self.max_tasks_per_gpu = max_tasks_per_gpu
        self.min_memory = min_memory
        self.gpu_task_counts = defaultdict(
            int
        )  # Tracks how many tasks are running on each GPU
        self.lock = threading.Lock()

    def execute_task_on_gpu(self, gpu_id, command):
        """
        Executes a given shell command on the specified GPU.

        Args:
            gpu_id (int): GPU index to run the task on.
            command (str): Shell command to execute.
        """

        def run():
            print(f"Executing task on GPU {gpu_id}: {command}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with self.lock:
                self.gpu_task_counts[gpu_id] += 1
            try:
                subprocess.run(command, shell=True, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"Task failed with error: {e}")
            finally:
                with self.lock:
                    self.gpu_task_counts[gpu_id] -= 1

        threading.Thread(target=run).start()

    def get_available_gpu(self):
        """
        Checks for an available GPU based on current usage and max task limits.

        Returns:
            int or None: ID of an available GPU, or None if all are busy.
        """
        try:
            # Get free and total memory for each GPU
            memory_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                text=True
            ).strip().splitlines()

            for line in memory_output:
                idx, free_mem, total_mem = map(int, line.strip().split(','))

                if isinstance(self.min_memory, float) and 0 < self.min_memory < 1:
                    required_mem = total_mem * self.min_memory
                else:
                    required_mem = self.min_memory

                with self.lock:
                    if (not self.allowed_gpu_ids or idx in self.allowed_gpu_ids) and self.gpu_task_counts[idx] < self.max_tasks_per_gpu and free_mem >= required_mem:
                        return idx
        except subprocess.CalledProcessError as e:
            print(f"Error checking GPU status: {e}")
        return None

    def run_tasks(self, tasks):
        """
        Runs a list of shell commands using available GPUs.

        Args:
            tasks (list[str]): List of shell command strings to execute.
        """
        task_index = 0
        while task_index < len(tasks):
            gpu_id = self.get_available_gpu()
            if gpu_id is not None:
                command = tasks[task_index]
                print(f"Found available GPU: {gpu_id} for task: {command}")
                self.execute_task_on_gpu(gpu_id, command)
                task_index += 1
            else:
                print("No available GPU or max task limit reached. Waiting...")
                time.sleep(self.wait_interval)
