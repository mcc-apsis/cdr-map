# Scientific Literature of CDR
This repository supports the publication *"Scientific literature on carbon dioxide removal revealed as much larger through AI-enhanced systematic mapping"*, available [here](https://www.researchsquare.com/article/rs-4109712/v1). The associated data can be explored and downloaded from [climateliterature.org](https://climateliterature.org/#/project/cdrmap).

The repository includes code for training and testing the classification models used to generate the categories in the systematic map, located in the [training](https://github.com/mcc-apsis/cdr-map/tree/main/training) folder. Additionally, it provides scripts for evaluating model performance and calculating confidence intervals, found in [check_models](https://github.com/mcc-apsis/cdr-map/tree/main/check_models). Finally, the code used to produce all published figures is available in [analysis_plots](https://github.com/mcc-apsis/cdr-map/tree/main/analysis_plots).

## Training and testing the classification models
The classification models are designed to categorize the titles and abstracts of scientific publications into predefined groups. To accomplish this, we use [ClimateBERT](https://huggingface.co/climatebert) — a transformer-based, pre-trained language model that has been fine-tuned to capture the domain-specific language typical of climate change literature — and further train it for our specific classification task.

### Installation Guide  
### How to use

