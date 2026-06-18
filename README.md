# Discrete Sky Operator (DiSkO) Synthesis Imaging

[![Build Status](https://travis-ci.org/tmolteno/disko.svg?branch=master)](https://travis-ci.org/tmolteno/disko)

Author: Tim Molteno tim@elec.ac.nz

Its so cool its POINTLESS. Image by using the telescope operator keeping track of the telescope null-space and range-space. DiSkO uses a discrete representation of the field of view (as a healpix grid, or unstructured mesh) in the sky space. This means we can image arbitratily wide fields of view (including the full sphere), as well as arbitrarily shaped fields (circles are preferred).

DiSkO can perform sparsity reduction by regularization and controls the volume of the sky solution. The result is an imaging algorithm that is sensitive to diffuse broad sources, and does not require restoration like CLEAN. Publications to appear :)

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

## TODO

* Add a --full-sphere option which fixes the sphere in celestial coordinates, and then points the phase center of an observation correctly. Requires a beam pattern to be specified (or at least a hemispherical beam). A beam is a sky vector mask. I.e., should fall to zero 'outside' the beam.
* Deal with flagging, output residuals in a way that can be used to flag in the measurement set. This means some casa expression that changes the MS.
