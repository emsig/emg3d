# -*- coding: utf-8 -*-
import os
import re
import sys
from setuptools import setup

if not sys.version_info[:2] >= (3, 7):
    sys.exit(f"emg3d is only meant for Python 3.7 and up.\n"
             f"Current version: {sys.version_info[0]}.{sys.version_info[1]}.")

# Get README and remove badges.
with open("README.rst") as f:
    readme = re.sub(r"\|.*\|", "", f.read(), flags=re.DOTALL)

setup(
    name="emg3d",
    description="A multigrid solver for 3D electromagnetic diffusion.",
    long_description=readme,
    author="The emsig community",
    author_email="info@emsig.xyz",
    url="https://emsig.xyz",
    license="Apache-2.0",
    packages=["emg3d", "emg3d.cli"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "emg3d=emg3d.cli.main:main",
        ],
    },
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.4",
        "numba>=0.45",
    ],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("emg3d", "version.py"),
    },
    setup_requires=["setuptools_scm"],
)
