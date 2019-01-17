#!/bin/sh

json_source_path=$1
pdf_dest_path=$2

# Save current directory to go back later.
currentdir=$(pwd)

# Create a temporary directory and save the filename.
tempdir=$(dirname $(mktemp -d))

# Get path to this script. 
# see: https://stackoverflow.com/a/1638397
# Absolute path to this script, e.g. /home/user/bin/foo.sh
script=$(readlink "$0")
# Absolute path this script is in, thus /home/user/bin
scriptpath=$(dirname "$script")

# Copy sqlplot directory to previously generated temporary directory.
cp -R $scriptpath/sqlplot $tempdir

cp $json_source_path $tempdir/sqlplot/plots.json

# Switch to coppied directory and generate pdf files with make.
cd $tempdir/sqlplot
make

# Move generated pdf files to destination.
mv *.pdf $pdf_dest_path

# Switch back to previously directory.
cd $currentdir

# Delete sqlplot directory in temporary directory.
rm -rf $tempdir/sqlplot