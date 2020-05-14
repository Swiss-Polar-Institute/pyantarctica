###############################################################
pyantarctica python package for the ACE-DATA and ASAID projects
###############################################################

ASAID / ACE-DATA: Data science for the antarctic
************************************************

Usage
*****

.. code-block:: python

    import pyantarctica
    from pyantarctica import aiceairsea 
    from pyantarctica import datafilter
    from pyantarctica import dataset
    from pyantarctica import visualizing
    from pyantarctica import windvectorcoordinates


Every notebook in the ACE-DATA data science project repository provides relatively simple (hopefully) and straightforward examples and scripts to run steps in the ASAID project analysis pipeline.

See scripts released with publications

Setup
*****

If you are using conda, first set up a conda environment:

(or pipenv works nicely as well)

.. code-block:: console

    $ conda create --name ace-data python=3.6
    $ conda activate ace-data

Install the pyanctarctica package
*********************************


From source:

.. code-block:: console

    $ cd YOUR_PROJECT_FOLDER
    $ mkdir src && cd src
    $ git clone git+git@github.com:Swiss-Polar-Institute/pyantarctica.git
    $ pip install ./pyantarctica

From the Swiss Polar Institute Github Repository https://github.com/Swiss-Polar-Institute/pyantarctica :

.. code-block:: console

    $ pip install -e git+git@github.com:Swiss-Polar-Institute/pyantarctica.git

Documentation
*************

Can be built with `sphinx`

Compatibility
**************

Python 3.3 preferred, but not required. Only tested with python >= 3

License
*******

Copyright 2017-2018 - Swiss Data Science Center (SDSC) and ACE-DATA/ASAID Project consortium. 

A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and Eidgenössische Technische Hochschule Zürich (ETHZ).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

Authors
*******

- Michele Volpi, SDSC, michele.volpi@sdsc.ethz.ch
- Sebastian Landwehr, PSI, sebastian.landwehr@psi.ch

Requirements
************

see `setup.py`
