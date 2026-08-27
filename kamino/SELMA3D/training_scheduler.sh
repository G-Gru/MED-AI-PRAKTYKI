#!/usr/bin/bash

config_directory="./configs"

# SIGUSR1: Kill active child process and continue loop
trap 'pkill -P $$' USR1

# SIGINT/SIGTERM: Kill active child process and exit script entirely
trap 'pkill -P $$; exit 1' INT TERM

for file in "$config_directory"/*
do
    python train.py "$file"
done