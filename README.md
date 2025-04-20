# GPU Task Scheduler

![](https://img.shields.io/pypi/pyversions/gpu_task_scheduler) ![](https://img.shields.io/pypi/v/gpu_task_scheduler) ![](https://img.shields.io/pypi/dm/gpu_task_scheduler) ![](https://img.shields.io/pypi/l/gpu_task_scheduler) ![](https://img.shields.io/github/stars/sangyx/gpu_task_scheduler?style=social)

A simple Python lib to schedule and execute deep learning tasks on available GPUs. It supports:

- Custom selection of GPUs.
- Maximum concurrency per GPU.
- Dynamic task dispatch using threading.

## 📦 Installation

```bash
pip install gpu_task_scheduler
```

## 🚀 Usage Example

```python
from gpu_task_scheduler import GPUTaskScheduler

tasks = [
    "python train_model.py --epochs 10",
    "bash run_simulation.sh",
    "python evaluate.py --model checkpoint.pth"
]

scheduler = GPUTaskScheduler(
    wait_interval=10,
    allowed_gpu_ids=[0, 1],
    max_tasks_per_gpu=2
)
scheduler.run_tasks(tasks)
```

## ⚙️ Parameters

### `GPUTaskScheduler(wait_interval=30, allowed_gpu_ids=None, max_tasks_per_gpu=1)`

| Parameter           | Type        | Default | Description |
|---------------------|-------------|---------|-------------|
| `wait_interval`     | `int`       | `30`    | Number of seconds to wait before checking again if no GPU is available. |
| `allowed_gpu_ids`   | `list[int]` | `None`  | List of GPU indices (e.g., `[0, 1]`) that the scheduler is allowed to use. If `None`, all GPUs are considered. |
| `max_tasks_per_gpu` | `int`       | `1`     | Maximum number of concurrent tasks allowed on a single GPU. Helps share GPU among multiple tasks. |

## 🛠️ Requirements

- Python 3.6+
- `nvidia-smi` available in PATH (NVIDIA drivers installed)