pyantarctica
==========

Umbrella package for ACE-DATA project. Data handling, modeling, visualization and (not much) more (for now)

Usage
-----

Installation
------------

from source

.. pip install -e --dev ../pyantartica

from the PyPI repos

.. pip install pyantarctica

Requirements
^^^^^^^^^^^^

see setup.py

Release process
^^^^^^^^^^^^^^^

1. Update version in `setup.py`, commit and tag.
2. Run `rm -rf dist && python setup.py sdist bdist_wheel && twine upload dist/*`
3. Change the bump the version to development (`X.Y.Z+1.devYYYYMMDD`)

Compatibility
-------------

License
-------

Authors
-------
