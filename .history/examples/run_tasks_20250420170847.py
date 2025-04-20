from gpu_task_scheduler import GPUTaskScheduler

tasks = [
    "python train_model.py --epochs 10",
    "bash run_simulation.sh",
    "python evaluate.py --model checkpoint.pth"
]

scheduler = GPUTaskScheduler(allowed_gpu_ids=[0, 1], max_tasks_per_gpu=2)
scheduler.run_tasks(tasks)
