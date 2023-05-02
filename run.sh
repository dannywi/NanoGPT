#!/bin/bash

# uncomment to download a file of your choice
# curl -o input.txt https://www.ccel.org/d/dostoevsky/crime/crime.txt

echo "====== RUNNING MODEL:" $(date) "======" | tee -a log.txt
time python bigram.py 2>&1 | tee -a log.txt
echo "====== FINISHED:" $(date) "======" | tee -a log.txt
