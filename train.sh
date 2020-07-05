#!/bin/bash -l
 
sbatch -p gpu --gres=gpu:k80:1 --mem=100g --time=1:00:00 run.sh

