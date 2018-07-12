# pyantarctica: ACE-DATA and ASAID projects python package

Package for ACE-DATA/ASAID. Data handling, run statistical models, visualization and (not much) more (for now)

## Usage

```python
import pyantarctica
import pyantarctica.dataset as data
import pyantarctica.modeling as mod
import pyantarctica.visualizing as viz
```

Every notebook in the ACE-DATA data science project repository provides relatively simple (hopefully) and straightforward examples and scripts to run steps in the ASAID project analysis pipeline.

See e.g. data preparation scrips `PROJ_XX_DataPreparation.ipynb` where XX is the subproject number.

## Installation

- from source:
  - `pip install -e --dev ../pyantarctica`

- from the PyPI repos:
  - `pip install pyantarctica`


## Requirements

see setup.py

## Release process

1. Update version in `setup.py`, commit and tag.
2. Run `rm -rf dist && python setup.py sdist bdist_wheel && twine upload dist/*`
3. Change the bump the version to development (`X.Y.Z+1.devYYYYMMDD`)

## Compatibility

## License

## Authors
