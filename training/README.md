## Training and testing the classification models
The classification models are designed to categorize the titles and abstracts of scientific publications into predefined groups. To accomplish this, we use [ClimateBERT](https://huggingface.co/climatebert) — a transformer-based, pre-trained language model that has been fine-tuned to capture the domain-specific language typical of climate change literature — and further train it for our specific classification task.

### Installation Guide 
#### Prerequisites 
- Up-to-date GNU/Linux operating system
- Python 3.10 or higher (with matching pip and virtualenv)
> **Note:** The code has not been tested on Windows.

#### Setup Instructions
After cloning the repository, set up the environment for training and testing the models:
'''
cd training/
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
'''

It is highly recommended to train the models on GPU-enabled machines to significantly reduce training time.

The models used for categorization in the publication were trained on the high-performance computing cluster of the Potsdam Institute for Climate Impact Research (PIK), utilizing their GPU environment.

To help users test and experiment with the scripts, we provide sample training datasets within the repository.

### How to use
There are two main scripts for training and testing the classifiers:

- One script trains a **binary classifier**, which is used to decide whether a document is relevant to our research question.
- The other script trains a **multiclass classifier**, which assigns relevant documents to specific categories based on their title and abstract.

Each script is controlled by parameters passed at runtime.

A script can operate in two modes controlled by the `testing` parameter:

- **Test mode**: In this mode, the training data is split into several parts, as defined by the `folds` parameter. A cross-validation procedure is performed where, in each round, one part is held out for testing while the remaining parts are used to train the model. This process repeats until each part has been used as test data once.

- **Training mode**: In this mode, the entire training dataset is used to train the final classification model, ensuring that no information is left out.

Instructions for setting parameters and switching between modes are provided in the script documentation. 




