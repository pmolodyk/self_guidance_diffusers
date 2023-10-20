#!/bin/bash
# Inria Dataset
# Download command: bash ./scripts/get_inria.sh

# Download/unzip
echo 'Downloading Inria...'
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
echo 'Moving Inria...'
tar xf inria.tar
mv INRIAPerson yolov7/data
rm inria.tar

wait # finish background tasks
