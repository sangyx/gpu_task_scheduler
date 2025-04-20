from setuptools import setup, find_packages

setup(
    name="gpu_task_scheduler",
    version="0.1.0",
    description="A python lib to schedule tasks on available GPUs.",
    author="sangyx",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
