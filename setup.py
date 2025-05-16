from setuptools import setup, find_packages

setup(
    name="gpu_task_scheduler",
    version="0.2.0",
    description="A python lib to schedule deep learning tasks on available GPUs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="sangyx",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
