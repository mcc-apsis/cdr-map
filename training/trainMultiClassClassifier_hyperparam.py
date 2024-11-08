import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from datetime import datetime

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from transformers import AutoModelForSequenceClassification, Trainer, logging

from torch import cuda
from torch import tensor
from torch import nn
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
parser.add_argument('--balanced', dest='balanced', action='store_true',
                    help='Flag if training should use balanced training data')
parser.add_argument('--no_balanced', dest='balanced', action='store_false',
                    help='Flag if training should use no balanced training data')
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
            data = pickle.load(handle)
        df = data[list(data.keys())[-1]].dropna(subset=["id"])
    if 'csv' in filename:
        df = pd.read_csv(filename)

    df = df.loc[(df[LABEL] == 0) | (df[LABEL] == 1)]

    df = df[["id", "category", FEATURE, LABEL]].copy()
    df["category"] = df.category.str.strip()
    allClasses = df.category.unique()

    df.drop_duplicates(subset=["id", "category", FEATURE], inplace=True)
    df = df.pivot(index=["id", FEATURE], columns='category', values=LABEL).reset_index()

    # get lines with at least one class label = only relevant documents
    df[allClasses] = df[allClasses].fillna(0)
    df["sumLabels"] = df[allClasses].apply(sum, axis=1)
    df = df.loc[df.sumLabels > 0].drop(columns=["sumLabels"])

    return df, allClasses


def loadCompleteData(filename):
    def createAllInfo(row):
        elements = ['title',
                    'author_keywords',
                    'abstract',
                    'document_type']
        string = ''
        for i in elements:
            try:
                if isinstance(row[i], str):
                    string += row[i]
                    string += '. '
            except KeyError:
                pass
        return string
    df = pd.read_csv(filename)
    df["all_info"] = df.apply(createAllInfo, axis=1)
    try:
        df = df.loc[df.pred_relevance >= 0.5, ['id', 'all_info', 'pred_relevance']]
    except AttributeError:
        df = df[['item_id', 'all_info']]
    return df


class ProbTrainer(Trainer):
    def __init__(self, balanced, *args, **kwargs):
        self.balanced = balanced
        super().__init__(*args, **kwargs)

    def predict_proba(self, test_dataset: Dataset) -> PredictionOutput:
        logits = self.predict(test_dataset).predictions
        if logits.shape[1] > 2:
            activation = Sigmoid()
        else:
            activation = Softmax()
        return activation(tensor(logits)).numpy()

    # used if balancing class weights
    def __calculate_weights(self):
        labels = self.train_dataset["label"]
        labels = pd.DataFrame(labels)

        weights = []
        for i in labels.columns:
            # weight_i = number_of_not_class/number_of_class
            try:
                weight_i = (len(labels) - sum(labels[i])) / sum(labels[i])
            except ZeroDivisionError:
                print('No samples in class')
                weight_i = (len(labels) - sum(labels[i])) / 1.
            weights.append(weight_i)
        return tensor(weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if cuda.is_available():
            labels.cuda()
            logits.cuda()

        if self.balanced:
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=self.__calculate_weights())
            if cuda.is_available():
                loss_fct.cuda()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)

        if not self.balanced:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

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
                          train_dataset=datasetify(train_df[FEATURE], tokenizer, train_df[allClasses].values),
                          eval_dataset=datasetify(test_df[FEATURE], tokenizer, test_df[allClasses].values),
                          balanced=args.balanced,
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
        return tokenizer(examples["text"], padding='longest', truncation=True)

    return dataset.map(tokenize_function, batched=True)


def getClassWithHighestProbability(row, allClasses):
    predClasses = ['pred_' + c for c in allClasses]
    predProbaClasses = ['predProba_' + c for c in allClasses]
    posClassification = sum(row[predClasses])
    if posClassification == 0:
        max_val = 0
        for predProbaClass in predProbaClasses:
            if row[predProbaClass] > max_val:
                max_val = row[predProbaClass]
                tech = predProbaClass.split("predProba_")[1]
        row["pred_" + tech] = 1
    return row


def getPredictionForTrainTest(df, trainer, allClasses, count):
    prediction = trainer.predict_proba(test_dataset=datasetify(df[FEATURE], tokenizer))
    pred_one_run = pd.DataFrame({'id': df["id"],
                                 'seed': [count for i in range(len(df))]}
                                )
    pred_one_run = pred_one_run.merge(df[["id"] + list(allClasses)], on='id', how='left')
    pred_one_run[['pred_' + c for c in allClasses]] = np.where(prediction >= 0.5, 1, 0)
    pred_one_run[['predProba_' + c for c in allClasses]] = prediction
    pred_one_run = pred_one_run.apply(lambda x: getClassWithHighestProbability(x, allClasses), axis=1)

    return pred_one_run


if __name__ == "__main__":
    args = parser.parse_args()

    print('testing:', args.testing)
    print('balanced:', args.balanced)
    print('hyperparemeter:', args.optimal_hyperparameter)
    print('folds:', FOLDS)

    df, allClasses = loadTrainingData(args.in_train)
    all_df = loadCompleteData(args.in_all)

    print(allClasses)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(allClasses))
    if cuda.is_available():
        model.cuda()

    if args.testing:
        all_test_runs = []
        all_train_runs = []

        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=43581)
        count = 0
        for train_index, test_index in kf.split(df):

            train, test = df.iloc[train_index], df.iloc[test_index]

            ## TODO for testing
            #train = train.head(20)
            #train_eval = train.tail(10)
            #test = test.head(10)

            print("Round", count)

            if args.optimal_hyperparameter:
                print("Hyperparameter search...")

                # eval is used to find optimal hyperparameter
                train_small, eval = train_test_split(train, test_size=0.1, random_state=43581)
                best_trial = find_best_trial_hyperparam(train_df=train_small, test_df=eval, model=model)

                best_trial_params = best_trial.hyperparameters
                with open(args.out_hyperparam, "a") as f:
                    f.write(f'run: {count}\n')
                    f.write(f'{best_trial_params}\n\n')

            print("\Training...")
            trainer = ProbTrainer(model=model,
                                  train_dataset=datasetify(train[FEATURE], tokenizer, train[allClasses].values),
                                  balanced=args.balanced,
                                  )
            if args.optimal_hyperparameter:
                trainer.train(trial=best_trial_params)
            else:
                trainer.train()

            print("\nEvaluating...")
            test_one_run = getPredictionForTrainTest(test, trainer, allClasses, count)
            train_one_run = getPredictionForTrainTest(train, trainer, allClasses, count)

            all_test_runs.append(test_one_run)
            all_train_runs.append(train_one_run)

            count += 1

        # store test result
        all_test_runs_df = pd.concat(all_test_runs)
        all_train_runs_df = pd.concat(all_train_runs)

        all_test_runs_df.to_csv(args.out_test, index=False)
        all_train_runs_df.to_csv(args.out_train, index=False)

    # use all training data for training and make predictions
    if not args.testing:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(allClasses))
        if cuda.is_available():
            model.cuda()

        if args.optimal_hyperparameter:
            print("Hyperparameter search...")

            # eval is used to find optimal hyperparameter
            train_small, eval = train_test_split(df, test_size=0.1, random_state=43581)
            best_trial = find_best_trial_hyperparam(train_df=train_small, test_df=eval, model=model)

            best_trial_params = best_trial.hyperparameters
            with open(args.out_hyperparam, "a") as f:
                f.write(f'{best_trial_params}\n\n')

        print("\Training...")
        trainer = ProbTrainer(model=model,
                              train_dataset=datasetify(df[FEATURE], tokenizer, df[allClasses].values),
                              balanced=args.balanced,
                              )
        if args.optimal_hyperparameter:
            trainer.train(trial=best_trial_params)
        else:
            trainer.train()

        if args.save_model:
            trainer.save_model(args.model_file)
            print("Model saved to ", args.model_file)

        print("\nPredicting...")
        train = getPredictionForTrainTest(df, trainer, allClasses, 1)
        train.to_csv(args.out_train, index=False)

        pred = trainer.predict_proba(test_dataset=datasetify(all_df[FEATURE], tokenizer))
        try:
            all_df = all_df[["id", FEATURE]]
        except KeyError:
            all_df = all_df[["item_id", FEATURE]]
        all_df[['pred_' + c for c in allClasses]] = np.where(pred >= 0.5, 1, 0)
        all_df[['predProba_' + c for c in allClasses]] = pred
        all_df = all_df.apply(lambda row: getClassWithHighestProbability(row, allClasses), axis=1)

        all_df.to_csv(args.out_all, index=False)

print("\n Done!")
