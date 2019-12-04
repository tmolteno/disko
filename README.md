# Discrete Sky Operator (DiSkO) Synthesis Imaging

[![Build Status](https://travis-ci.org/tmolteno/disko.svg?branch=master)](https://travis-ci.org/tmolteno/disko)

Author: Tim Molteno tim@elec.ac.nz

Its so cool its gridless. Image by using the telescope operator keeping track of the telescope null-space and range-space.

## Howto

    disko --display --show-sources

To load a data from a measurement set 

    disko --ms test_data/test.ms --tikhonov --nside 32 --PDF
    
## Changelog

0.6.0  Get data from Measurement Sets!
0.5.0b5 Allow sources not to be shown.
0.5.0b4 Override plot in HPSubSphere to allow for non-normal pixels.
0.5.0b3 Added elliptical source circle projections in SVG.
0.5.0 Getting imaging logic better. Added L2 regularization, and cross-validation
