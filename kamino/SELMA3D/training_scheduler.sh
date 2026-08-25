#!/usr/bin/bash

config_directory="./configs"

for file in "$config_directory"/* 
do
    python train.py "$file"
done
s