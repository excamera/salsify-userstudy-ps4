#!/bin/bash

./delay.py
./quality.py

./svg2pdf

cp delay.pdf ../../salsify-paper/figures/userstudy-ps4-delay.pdf
cp quality.pdf ../../salsify-paper/figures/userstudy-ps4-quality.pdf 

cd ../../salsify-paper/figures

rm userstudy-ps4-delay-crop.pdf
rm userstudy-ps4-delay-quality.pdf

pdfcrop userstudy-ps4-delay.pdf
pdfcrop userstudy-ps4-quality.pdf

make -C ..
