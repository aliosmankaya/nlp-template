# NLP Template

***Customizable NLP Model Development Template***

This customizable template for natural language processing (NLP) tasks. With simple modifications in a single config.py file, it enables the development of various NLP models. The primary goal is to offer a starting point for developers working in the NLP domain.

## Usage

Check the example usage of **nlp-template** on this [notebook](https://www.kaggle.com/code/aliosmankaya/email-classification-with-nlp-template/notebook)

## Install

Clone the project to your local environment:

```
git clone https://github.com/aliosmankaya/nlp-template.git
```

Install the necessary dependencies:

```
pip install -r requirements.txt
```

## Data Format

* Data must be a CSV file
* Data must have 2 columns:
    * **text**: Text body (*str*)
    * **labels**: Text labels (*int*)


## Customize

* **task:** Problem task (*str*, defaults to "binary")

* **num_labels**: Number of labels (*int*, defaults to 2)
* **model_path**: Pretrained model path (*str*, defaults to "bert-base-uncased")
* **epochs**: Epochs number (*int*, defaults to 3)
* **batch_size**: Batch size (*int*, defaults to 8)
* **max_length**: Maximum length of the sequence (*int*, defaults to 200)
* **learning_rate**: Learning rate (*float*, defaults to 5e-5)
* **seed**: Seed number (*int*, defaults to 42)
* **device**: Device type (*str*, defaults "cuda" or "cpu")
* **val_strategy**: Validation strategy (*func*, defaults to StratifiedKFold)
* **n_splits**: Fold number (*int*, defaults to 5)
* **metric**: Metric (*func*, defaults to F1Score)

## Train

To initiate the training phase, use the following command:

```
python3 train.py --path PATH
```

* **path** : Train data path


## Inference

Coming soon...
