#!/usr/bin/env bash

conda init 
conda env create --file env.yml
source activate env