from setuptools import setup, find_packages

setup(
    name="kernel_zoo",
    packages=find_packages(),  # Automatically find packages in your source directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch==2.3.1",
        "pre-commit==3.8.0",
        "numba==0.60.0",
    ]
)
