#!/bin/sh

OPTS="--mu --var --pcf --PNG --nside 20 --posterior post.h5 --title seq --dir seq_out --sigma-v=0.15"

# First convert the hdf file to a sequence of measurment sets.
# tart2ms --hdf vis_2021-03-25_20_50_23.568474.hdf

DIR=./test_data/
FIRST=$(find $DIR -name 'tart.ms_*' | sort | head -n 1)

disko_bayes --ms $FIRST $OPTS

for ms in $(find $DIR -name 'tart.ms_*' | sort | tail -n +2)
do
    disko_bayes --ms $ms --prior post.h5 $OPTS
done
