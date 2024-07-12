#!/bin/bash

#SBATCH --nodes=1
#SBATCH -o ai4eo_runner.%J.out
#SBATCH --time=24:00:00
#SBATCH --mem=250000
#SBATCH --ntasks-per-node=128
#SBATCH -A climate
#SBATCH --partition=milan


source ~/miniconda3/bin/activate py311

for idx in $(seq 45 49)
    do
        python make_data.py \
        --idx=$idx \
        --year=2010 \
        --path_data=/nvme/h/pgeorgiades/data_p143/UFPs/dat3/ML_paper2/scripts_for_paper_testground \
        --dest=/nvme/h/pgeorgiades/data_p143/UFPs/dat3/ML_paper2/scripts_for_paper/data/inference_data
    done

python make_data.py \
    --idx=39 \
    --year=2010 \
    --path_data=/nvme/h/pgeorgiades/data_p143/UFPs/dat3/ML_paper2/scripts_for_paper_testground \
    --dest=/nvme/h/pgeorgiades/data_p143/UFPs/dat3/ML_paper2/scripts_for_paper/data/inference_data