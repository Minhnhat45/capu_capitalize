#! /bin/sh
#
# run_prepare_data.sh
# Copyright (C) 2022 nhat-l <nhat-l@rd04>
#
# Distributed under terms of the MIT license.
#

FILE_NAME="data-bin-local"

if [ -d $FILE_NAME ];
then
    mkdir data-bin-local
    mkdir data-bin-local/vlsp
    mkdir data-bin-local/vlsp/preprocessed
    python prepare_data.py
else
    echo "$FILE_NAME already exist, prepare_data is exporting processed data to that directory";
    python prepare_data.py
fi
