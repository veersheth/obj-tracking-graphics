#!/bin/bash

file=$1

rm -rf 01_images
mkdir 01_images

# split the video up
ffmpeg -i "$file" 01_images/img%04d.png

# sift
source env/bin/activate
python create_sifts.py

# combine sifted images
ffmpeg -framerate 24 -i 02_sifting/img%04d.png -c:v mjpeg out.avi
