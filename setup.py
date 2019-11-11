#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

from setuptools import setup, find_packages

import setuptools.command.test

with open('README.md') as f:
    readme = f.read()

setup(name='disko',
    version='0.5.0b4',
    description='Discrete Sky Operator (DiSkO) Aperture Synthesis Radio Imaging',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='http://github.com/tmolteno/TART',
    author='Tim Molteno',
    test_suite='nose.collector',
    tests_require=['nose'],
    author_email='tim@elec.ac.nz',
    license='GPLv3',
    install_requires=['numpy', 'matplotlib', 'healpy', 'astropy', 'tart', 'tart-tools', 'h5py', 'scipy', 'svgwrite', 'dask', 'scikit-learn'],
    packages=['disko'],
    scripts=['bin/disko', 'bin/disko_svd'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications :: Ham Radio",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Intended Audience :: Science/Research"])
