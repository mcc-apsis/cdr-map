#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=6000
#SBATCH --qos=gpushort

#SBATCH --job-name=relevant_class_distil
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.errq
#SBATCH --workdir=/home/salueck/nets_climatebert


today=$(date '+%Y-%m-%d')

project_dir="/home/salueck/nets_climatebert"
#project_dir="/home/sarah/projects/nets_climatebert"

in_train="$project_dir/data/2022-12-09_relevant_training.csv"
in_all="$project_dir/data/wos_scopus_all.csv"
out_test="$project_dir/data/"$today"_test_predicted_relevant.csv"
out_all="$project_dir/data/"$today"_relevant_predicted.csv"
model="distilroberta-base"

echo '$out_test'

module load anaconda/5.0.0_py3
source activate venv
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python trainRelevantClassifier.py \
--in_train $in_train \
--in_all $in_all \
--out_test $out_test \
--out_all $out_all \
--model $model