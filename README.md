# Discrete Sky Operator (DiSkO) Synthesis Imaging

[![Build Status](https://travis-ci.org/tmolteno/disko.svg?branch=master)](https://travis-ci.org/tmolteno/disko)

Author: Tim Molteno tim@elec.ac.nz

Its so cool its POINTLESS. Image by using the telescope operator keeping track of the telescope null-space and range-space. This software 
carries out a sparsity reduction by regularization and controls the volume of the sky solution. The result is an imaging algorithm that is sensitive to diffuse broad sources, and does not require restoration like CLEAN. Publications to appear :)

## Howto

    disko --display --show-sources

To load a data from a measurement set 

    disko --ms test_data/test.ms --tikhonov --nside 32 --PDF

## VLA imaging with DiSkO

Download the VLA 5GHZ continuum survey measurement set. AG733_A061209.xp1 from the NRAO site.

Calibrate and then split the measurement set, following the CASA tutorial [https://casaguides.nrao.edu/index.php/VLA_5_GHz_continuum_survey_of_Seyfert_galaxies]

    disko --fov 0.01 --ms NGC1194.split.ms --SVG --arcmin 0.0025 --tikhonov
## More challenging

This tutorial should generate a file with lots of diffuse radiation. 
[https://casaguides.nrao.edu/index.php/VLA_Continuum_Tutorial_3C391-CASA5.5.0]

    wget http://casa.nrao.edu/Data/EVLA/3C391/3c391_ctm_mosaic_10s_spw0.ms.tgz
    wget https://github.com/jaredcrossley/CASA-Guides-Script-Extractor/blob/master/extractCASAscript.py
    
    python extractCASAscript.py 'https://casaguides.nrao.edu/index.php?title=VLA_Continuum_Tutorial_3C391-CASA5.5.0'

Then in CASA

    execfile("VLAContinuumTutorial3C391-CASA5.5.0.py")
    
This should generate a suitable measurement set to image.

## Changelog

0.6.0b9 Report Nyquist resolution
0.6.0b7 MS were being read incorrectly - the UVW are measured in meters, not wavelengths!
0.6.0b6 Correct field pointing from measurement sets.
0.6.0b5 Reduce memory requirements by around 25%.
0.6.0b4 Report the r^2 value.
0.6.0b2  Use dask for very large jobs (use the --dask switch)
0.6.0b1  Get data from Measurement Sets!
0.5.0b5 Allow sources not to be shown.
0.5.0b4 Override plot in HPSubSphere to allow for non-normal pixels.
0.5.0b3 Added elliptical source circle projections in SVG.
0.5.0 Getting imaging logic better. Added L2 regularization, and cross-validation
