#!/bin/bash

mkdir resampled


ffmpeg -i 20_074.mp4 -vf scale=342:256 -strict -2 20_074_2.mp4 -hide_banner
ffmpeg -i 20_074_2.mp4 -vf fps=10 img%06d.jpg -hide_banner
