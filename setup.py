from setuptools import find_packages, setup

install_requires = [req.strip() for req in "numpy, scipy, pandas, jupyter, seaborn, matplotlib".split(',')]

tests_require = []

extras_require = {
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
