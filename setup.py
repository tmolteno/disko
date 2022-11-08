#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

setup(name='disko',
      version='0.9.4b1',
      description='Discrete Sky Operator (DiSkO) Aperture Synthesis Radio Imaging',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='http://github.com/tmolteno/disko',
      author='Tim Molteno',
      test_suite='nose.collector',
      tests_require=['nose'],
      author_email='tim@elec.ac.nz',
      license='GPLv3',
      install_requires=['numpy', 'matplotlib', 'healpy', 'astropy', 'tart', 'tart-tools', 'tart2ms', 'h5py',
                        'scipy', 'svgwrite', 'dask', 'scikit-learn', 'dask-ms', 'distributed',
                        'pylops', 'toolz', 'dmsh', 'optimesh', 'imageio'],
      packages=['disko'],
      scripts=['bin/disko', 'bin/disko_svd', 'bin/disko_bayes'],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering",
          "Topic :: Communications :: Ham Radio",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3 :: Only',
          "Intended Audience :: Science/Research"])
