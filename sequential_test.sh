#!/bin/sh

DIR=../tart2ms/
FIRST=$(find $DIR -name 'test.ms_*' | sort | head -n 1)

disko_bayes --ms $FIRST --mu --PNG --nside 20 --posterior post.h5 --title 'seq' --dir seq_out

for ms in $(find $DIR -name 'test.ms_*' | sort | tail -n +2)
do
    disko_bayes --ms $ms --prior post.h5 --mu --pcf --var --PNG --nside 20 --posterior post.h5 --title 'seq' --dir seq_out
done
