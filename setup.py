#
# Copyright Tim Molteno 2019-2024 tim@elec.ac.nz
#

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

setup(name='disko',
      version='1.0.0b4',
      description='Discrete Sky Operator (DiSkO) Aperture Synthesis Radio Imaging',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='http://github.com/tmolteno/disko',
      author='Tim Molteno',
      test_suite='nose.collector',
      tests_require=['nose'],
      author_email='tim@elec.ac.nz',
      license='GPLv3',
      install_requires=['numpy', 'matplotlib', 'healpy', 'astropy', 'h5py',
                        'scipy', 'svgwrite', 'dask', 'scikit-learn', 'distributed',
                        'pylops', 'toolz', 'pygmsh', 'imageio'],
      extras_require = {
        'tart':  ['tart', 'tart_tools', 'tart2ms' ]
      },
      packages=['disko', 'disko.fov'],
      scripts=['bin/disko', 'bin/disko_svd', 'bin/disko_bayes', 'bin/disko_draw'],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering",
          "Topic :: Communications :: Ham Radio",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3 :: Only',
          "Intended Audience :: Science/Research"])
