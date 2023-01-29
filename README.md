# enzymeNER
The enzymeNER contains two deep learning based models for named-entity recognition (NER) of enzyme which can be trained on the provided corpus for enzyme extraction from inputs. A newly designed Annotation Pipeline is also proposed for automatically labelling  enzyme-entities on training corpus. 

## Installation dependence
| Name | Version |
|------|---------|
|numpy|1.19.5|
|python|3.8.3|
|tensorflow|2.3.0|
|tokenizers|0.11.4|

## Train and evaluate the model
To Train the model, run 'run_train.py' to start training with the operation inside changing to 'train'. And when the operation is changed to 'test', this file will execute the evaluation process and The metrics including Precision, Recall and F1-score will output as the result.


enzymeNER provides
## Infer the model
