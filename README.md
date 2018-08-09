# pyantarctica: ACE-DATA and ASAID projects python package

Package for ACE-DATA/ASAID. Data handling, run statistical models, visualization and (not much) more (for now)

## Usage

```python
import pyantarctica
import pyantarctica.dataset as data
import pyantarctica.modeling as mod
import pyantarctica.visualizing as viz
```

Every notebook in the ACE-DATA data science project repository provides relatively simple (hopefully) and straightforward examples and scripts to run steps in the ASAID project analysis pipeline.

See e.g. data preparation scrips `PROJ_XX_DataPreparation.ipynb` where XX is the subproject number.

## Installation

- from source:
  - `pip install -e pyantarctica_folder`

- from the PyPI repos:
  - `pip install pyantarctica`

- from the GitLab:
  - `pip install -e git+https://gitlab.datascience.ch/ACE-ASAID/pyantarctica.git#egg=pyantarctica`
  - and editable clone of the repository will be available locally
  - * access required

## Release process

1. Update version in `setup.py`, commit and tag.
2. Run `rm -rf dist && python setup.py sdist bdist_wheel && twine upload dist/*`
3. Change the bump of the version to development (`X.Y.Z+1.devYYYYMMDD`)

## Compatibility

Python 3 preferred, but not required.

## License

Copyright 2017-2018 - Swiss Data Science Center (SDSC)

A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and Eidgenössische Technische Hochschule Zürich (ETHZ).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

## Authors

- Michele Volpi, SDSC, michele.volpi@sdsc.ethz.ch
- Sebastian Landwehr, PSI, sebastian.landwehr@psi.ch

## Requirements

Working with following extra packages:

```python
    install_requires = [
        'Cartopy==0.16.0'
        'GPy==1.9.2'
        'matplotlib==2.2.2'
        'numpy==1.14.5'
        'pandas==0.23.0'
        'pickleshare==0.7.4'
        'scikit-learn==0.19.1'
        ]
```
