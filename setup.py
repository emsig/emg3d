# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup

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
        "Programming Language :: Python",
    ],
    entry_points={
        "console_scripts": [
            "emg3d=emg3d.cli.main:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "scipy>=1.9",
        "numba>=0.53",
    ],
    extras_require={
        'full': [
            'tqdm',
            'h5py',
            'scooby',
            'xarray',
            'empymod',
            'discretize>=0.7.3',
            'matplotlib',
        ],
    },
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("emg3d", "version.py"),
    },
    setup_requires=["setuptools_scm"],
)
