#!/bin/bash

mplayer $1 -demuxer rawvideo -rawvideo h=720:w=1280:format=bgra:fps=1
