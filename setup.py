# -*- coding: utf-8 -*-
import os
import re
import sys
from setuptools import setup

if not sys.version_info[:2] >= (3, 7):
    sys.exit(f"emg3d is only meant for Python 3.7 and up.\n"
             f"Current version: {sys.version_info[0]}.{sys.version_info[1]}.")

# Get README and remove badges.
readme = open('README.rst').read()
readme = re.sub('----.*marker', '----', readme, flags=re.DOTALL)

description = 'A multigrid solver for 3D electromagnetic diffusion.'

setup(
    name='emg3d',
    description=description,
    long_description=readme,
    author='The emg3d Developers',
    author_email='dieter@werthmuller.org',
    url='https://emsig.github.io',
    license='Apache License V2.0',
    packages=['emg3d', 'emg3d.cli'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'emg3d=emg3d.cli.main:main',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'scipy>=1.4.0',
        'numba>=0.45.0',
    ],
    use_scm_version={
        'root': '.',
        'relative_to': __file__,
        'write_to': os.path.join('emg3d', 'version.py'),
    },
    setup_requires=['setuptools_scm'],
)
