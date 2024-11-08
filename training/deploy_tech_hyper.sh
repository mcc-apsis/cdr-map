#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000
#SBATCH --qos=gpushort

#SBATCH --job-name=tech_class_test_hyperparam
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.errq
#SBATCH --workdir=/home/salueck/nets_climatebert

date=$(date '+%Y-%m-%d')

project_dir="/home/salueck/nets_climatebert"
project_dir="/p/tmp/salueck/nets_climatebert"

model="climatebert/distilroberta-base-climate-f"

#project_dir="/home/sarah/projects/nets_climatebert"
#model="distilroberta-base"

in_train="$project_dir/data/2022-12-06_technologies_training.csv"
in_all="$project_dir/data/2023-04-11_relevant_predicted_Hyperparam.csv"
out_hyperparam="$project_dir/data/"$date"_tech_hyperparam_results.txt"
out_test="$project_dir/data/"$date"_test_predicted_tech_unbalanced_hyperparamTuning.csv"
out_train="$project_dir/data/"$date"_train_predicted_tech_unbalanced_hyperparamTuning.csv"
out_all="$project_dir/data/"$date"_tech_predicted_unbalanced.csv"
balanced="no_balanced" # balanced / no_balanced
optimal_hyperparameter="optimal_hyperparameter" # optimal_hyperparameter / no_optimal_hyperparameter
testing="testing" # testing / no_testing
save_model="save_model" # save_model / no_save_model
model_file="$project_dir/"$date_"tech_unbalanced_hyperparam_model"
folds=3

touch $out_hyperparam
module load anaconda/5.0.0_py3
module load git/2.5.0
source activate venv
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python trainMultiClassClassifier_hyperparam.py \
--in_train $in_train \
--in_all $in_all \
--out_hyperparam $out_hyperparam \
--out_test $out_test \
--out_train $out_train \
--out_all $out_all \
--model_name $model \
--$balanced \
--$optimal_hyperparameter \
--$testing \
--$save_model \
--model_file $model_file \
--folds $folds