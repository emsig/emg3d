# -*- coding: utf-8 -*-
import os
from setuptools import setup

readme = open('README.rst').read()

description = 'A multigrid solver for 3D electromagnetic diffusion.'

setup(
    name='emg3d',
    description=description,
    long_description=readme,
    author='The emg3d Developers',
    author_email='dieter@werthmuller.org',
    url='https://empymod.github.io',
    license='Apache License V2.0',
    packages=['emg3d'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy>=1.15.0',
        'scipy>=1.1.0',
        'numba>=0.40.0',
    ],
    setup_requires=[
        'setuptools_scm'
    ],
    use_scm_version=dict(
        root = '.',
        relative_to = __file__,
        write_to = os.path.join(os.path.dirname(__file__), 'emg3d/version.py')
    ),
)
