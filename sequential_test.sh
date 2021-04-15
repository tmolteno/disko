#!/bin/sh

OPTS="--mu --PNG --nside 20 --posterior post.h5 --title seq --dir seq_out --sigma-v=0.15"

DIR=../tart2ms/
FIRST=$(find $DIR -name 'test.ms_*' | sort | head -n 1)

disko_bayes --ms $FIRST $OPTS

for ms in $(find $DIR -name 'test.ms_*' | sort | tail -n +2)
do
    disko_bayes --ms $ms --prior post.h5 $OPTS
done
