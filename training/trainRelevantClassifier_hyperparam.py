import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import argparse

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from transformers import AutoModelForSequenceClassification, Trainer, logging

from torch import tensor
from torch.nn import Sigmoid, Softmax
from transformers.trainer_utils import PredictionOutput
from ray import tune

logging.set_verbosity_warning()

LABEL = "label_3"
FEATURE = "all_info"

parser = argparse.ArgumentParser()
parser.add_argument('--in_train', dest='in_train', type=str,
                    help='Add filename for training data')
parser.add_argument('--in_all', dest='in_all', type=str,
                    help='Add filename for complete dataset')
parser.add_argument('--out_hyperparam', dest='out_hyperparam', type=str,
                    help='Add filename for storage of test runs during hyperaram search.')
parser.add_argument('--out_test', dest='out_test', type=str,
                    help='Add filename for test dataset')
parser.add_argument('--out_train', dest='out_train', type=str,
                    help='Add filename for prediction on training data')
parser.add_argument('--out_all', dest='out_all', type=str,
                    help='Add filename for complete prediction')
parser.add_argument('--model_name', dest='model_name', type=str,
                    help='Add filename for model')
parser.add_argument('--optimal_hyperparameter', dest='optimal_hyperparameter', action='store_true',
                    help='Flag if hyperparameter should be optimized')
parser.add_argument('--no_optimal_hyperparameter', dest='optimal_hyperparameter', action='store_false',
                    help='Flag if hyperparameter should not be optimized')
parser.add_argument('--testing', dest='testing', action='store_true',
                    help='Flag if training procedure should be tested.'
                    )
parser.add_argument('--no_testing', dest='testing', action='store_false',
                    help='Flag if training procedure should not be tested.'
                    )
parser.add_argument('--save_model', dest='save_model', action='store_true',
                    help='Flag if trained model should be saved.'
                    )
parser.add_argument('--no_save_model', dest='save_model', action='store_false',
                    help='Flag if trained model should not be saved.'
                    )
parser.add_argument('--model_file', dest='model_file', type=str,
                    help='File name of trained model'
                    )
parser.add_argument('--folds', dest='folds', type=int, nargs='?', const=4, default=4,
                    help='Number of folds for cross-validation. The complete dataset is used.'
                    )

FOLDS = parser.parse_args().folds


def loadTrainingData(filename):
    if 'pickle' in filename:
        with open(filename, 'rb') as handle:
            rel = pickle.load(handle)
        df_rel = rel[list(rel.keys())[-1]]
    if 'csv' in filename:
        df_rel = pd.read_csv(filename)

    df_rel = df_rel.loc[(df_rel[LABEL] == 0) | (df_rel[LABEL] == 1)]
    df_rel.drop_duplicates(subset=['id'], inplace=True)
    return df_rel


def loadCompleteData(filename):
    def createAllInfo(row):
        elements = [
            'title',
            #'author_keywords',
            'abstract',
            #'document_type'
        ]
        string = ''
        for i in elements:
            if isinstance(row[i], str):
                string += row[i]
                string += '. '
        return string

    all_df = pd.read_csv(filename).drop_duplicates()
    all_df["all_info"] = all_df.apply(createAllInfo, axis=1)
    return all_df


class ProbTrainer(Trainer):
    def predict_proba(self, test_dataset: Dataset) -> PredictionOutput:
        logits = self.predict(test_dataset).predictions
        if logits.shape[1] > 2:
            activation = Sigmoid()
        else:
            activation = Softmax()
        return activation(tensor(logits)).numpy()

def find_best_trial_hyperparam(train_df, test_df, model):
    training_args = TrainingArguments(
        #"test"
        evaluation_strategy="steps",
        eval_steps=1e9,
        disable_tqdm=True,
        output_dir="/p/tmp/salueck/tmp_trainer"
    )

    def model_init(trial):
        return model

    trainer = ProbTrainer(model=None,
                          train_dataset=datasetify(train_df[FEATURE], tokenizer, train_df[LABEL].values),
                          eval_dataset=datasetify(test_df[FEATURE], tokenizer, test_df[LABEL].values),
                          model_init=model_init,
                          args=training_args,
                          )

    def ray_hp_space(trial):
        return {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
            "weight_decay": tune.uniform(0, 0.3),
            "warmup_steps": tune.loguniform(1, 500),
            "num_epochs": tune.choice([2, 3, 4, 5])
        }

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=ray_hp_space,
        backend="ray",
        n_trials=20,  # number of trials
        local_dir="/p/tmp/salueck/ray_log/",
        keep_checkpoints_num=1,
        resources_per_trial={"cpu": 14, "gpu": 1},
    )

    return best_trial

# tokenizing
def datasetify(x, tokenizer, y=None):
    data_dict = {"text": x}
    if y is not None:
        data_dict["label"] = y
    dataset = Dataset.from_dict(data_dict)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="longest", truncation=True)

    return dataset.map(tokenize_function, batched=True)

def getPredictionForTrainTest(df, trainer, count):
    prediction = trainer.predict_proba(test_dataset=datasetify(df[FEATURE], tokenizer))
    pred_one_run = pd.DataFrame({'id': df["id"],
                                 'seed': [count for i in range(len(df))]}
                                )
    pred_one_run = pred_one_run.merge(df[["id", LABEL]], on='id', how='left')
    pred_one_run["predProba"] = prediction[:, 1]
    pred_one_run["pred"] = np.where(prediction[:, 1] >= 0.5, 1, 0)

    return pred_one_run


if __name__ == "__main__":
    args = parser.parse_args()

    print('testing:', args.testing)
    print('hyperparemeter:', args.optimal_hyperparameter)
    print('folds:', FOLDS)

    df = loadTrainingData(args.in_train)
    all_df = loadCompleteData(args.in_all)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    # test training process, no prediction
    testing = True
    if args.testing:
        test_all_runs = []
        train_all_runs = []

        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=43581)
        count = 0
        for train_index, test_index in kf.split(df):
            print("Round", count)
            # preparing data
            train, test = df.iloc[train_index], df.iloc[test_index]

            ## TODO for testing
            #train = train.head(20)
            #test = test.head(10)
            #all_df = all_df.head(20)
            if args.optimal_hyperparameter:
                print("Hyperparameter search...")

                # eval is used to find optimal hyperparameter
                train_small, eval = train_test_split(train, test_size=0.1, random_state=43581)
                best_trial = find_best_trial_hyperparam(train_df=train_small, test_df=eval, model=model)

                best_trial_params = best_trial.hyperparameters
                with open(args.out_hyperparam, "a") as f:
                    f.write(f'run: {count}\n')
                    f.write(f'{best_trial_params}\n\n')

            print("Training...")
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
            trainer = ProbTrainer(model=model, train_dataset=datasetify(train[FEATURE], tokenizer, train[LABEL]))
            if args.optimal_hyperparameter:
                trainer.train(trial=best_trial_params)
            else:
                trainer.train()

            print("Evaluating...")
            test_one_run = getPredictionForTrainTest(test, trainer, count)
            train_one_run = getPredictionForTrainTest(train, trainer, count)

            test_all_runs.append(test_one_run)
            train_all_runs.append(train_one_run)

            count += 1

        # store test result
        all_test_runs_df = pd.concat(test_all_runs)
        all_test_runs_df.to_csv(args.out_test, index=False)

        all_train_runs_df = pd.concat(train_all_runs)
        all_train_runs_df.to_csv(args.out_train, index=False)

    # use all training data and make predictions:
    if not args.testing:

        print("Training...")
        if args.optimal_hyperparameter:
            print("Hyperparameter search...")

            # eval is used to find optimal hyperparameter
            train_small, eval = train_test_split(df, test_size=0.1, random_state=43581)
            best_trial = find_best_trial_hyperparam(train_df=train_small, test_df=eval, model=model)

            best_trial_params = best_trial.hyperparameters
            with open(args.out_hyperparam, "a") as f:
                f.write(f'{best_trial_params}\n\n')

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
        trainer = ProbTrainer(model=model,
                              train_dataset=datasetify(df[FEATURE], tokenizer, df[LABEL])
                              )
        if args.optimal_hyperparameter:
            trainer.train(trial=best_trial_params)
        else:
            trainer.train()

        if args.save_model:
            trainer.save_model(args.model_file)
            print("Model saved to ", args.model_file)

        print("Make predictions...")
        train_pred = trainer.predict_proba(test_dataset=datasetify(df[FEATURE], tokenizer))
        df["pred_relevance"] = train_pred[:, 1]

        df.to_csv(args.out_train, index=False)
        all_df_pred = trainer.predict_proba(test_dataset=datasetify(all_df[FEATURE], tokenizer))
        all_df["pred_relevance"] = all_df_pred[:, 1]
        all_df.to_csv(args.out_all, index=False)

    print("\nDone!")
