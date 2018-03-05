from setuptools import find_packages, setup

install_requires = [req.strip() for req in "numpy, scipy, pandas, jupyter, seaborn, matplotlib".split(',')]

tests_require = [
    'check-manifest>=0.25',
    'coverage>=4.0',
    'isort>=4.2.2',
    'pydocstyle>=1.0.0',
    'pytest-cache>=1.0',
    'pytest-cov>=1.8.0',
    'pytest-pep8>=1.0.6',
    'pytest>=2.8.0',
]

extras_require = {
    'docs': [
        'Sphinx>=1.5.1',
    ],
    'tests': tests_require,
}

setup(
    name="ace-package",
    version="0.1.0",
    url="",

    author="Michele Volpi, Swiss Data Science Center (SDSC)",
    author_email="michele.volpi@datascience.ch",

    description="Umbrella package for ACE-DATA (sub-)project(s).",
    long_description=open('README.rst').read(),

    packages=find_packages(),
    zip_safe=True,

    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=['pytest'],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)
