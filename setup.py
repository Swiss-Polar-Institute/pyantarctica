# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 - Swiss Data Science Center (SDSC) and ACE-DATA/ASAID Project consortium.
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ). Written within the scope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

# Manually install:
# 'Cartopy==0.16.0',
# 'geoviews',

install_requires = ["matplotlib", "numpy", "pandas", "airsea", "pathlib"]

tests_require = []

extras_require = {}

setup(
    name="pyantarctica",
    version="0.1.0",
    license="Apache License 2.0",
    url="",
    author="Michele Volpi, Swiss Data Science Center (SDSC, ETHZ and EPFL); Sebastian Landwehr (Extreme Environments Research Laboratory, EPFL)",
    author_email="michele.volpi@datascience.ch",
    description="Umbrella package for ACE-DATA ASAID (sub-)project(s).",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=["pytest"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
