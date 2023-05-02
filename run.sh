#!/bin/bash

# uncomment to download a file of your choice
# curl -o input.txt https://www.ccel.org/d/dostoevsky/crime/crime.txt

current_date_time=$(date)
echo "====== RUNNING MODEL: $current_date_time ======" | tee -a log.txt
time python bigram.py | tee -a log.txt
