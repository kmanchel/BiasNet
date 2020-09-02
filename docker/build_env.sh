#!/usr/bin/env bash

conda init 
conda env create --file environment.yml
source activate env