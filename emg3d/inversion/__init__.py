"""
The inversion submodule of emg3d provides wrapper functionalities to use emg3d
as a forward modelling kernel within third-party inversion frameworks. These
third-party libraries are not included in emg3d, you have to install them
Separately.

Currently implemented wrappers and their corresponding requirements:

- pyGIMLi(emg3d): Requires the *Geophysical Inversion & Modelling
  Library* `pyGIMLi <https://pygimli.org>`_.
"""
# Copyright 2024 The emsig community.
#
# This file is part of emg3d.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
import importlib as _importlib


submodules = [
    'pygimli',
]

__all__ = submodules


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"emg3d.inversion.{name}")
