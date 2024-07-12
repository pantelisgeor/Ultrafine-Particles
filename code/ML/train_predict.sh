#!/bin/bash

path_dat_train=$1
path_model=$2
scaler_feats=$3
path_dat_inf=$4
model_name=MAPIE

# First train the model
python train_model.py \
    --path_data=$path_dat_train \
    --path_model=$path_model \
    --model_name=$model_name \
    --scaler_feats=$scaler_feats

# Then run inference for all the years 2010-2019
for year in $(seq 2010 2019)
    do
        python predict_year.py \
            --year=$year \
            --path_data=$path_dat_inf \
            --path_model=$path_model \
            --model_name=$model_name \
            --scaler_feats=$scaler_feats
    done
