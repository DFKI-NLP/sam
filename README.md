# SAM @SciArg
This is the official implementation of our paper accepted at [WIESP22](https://www.aclweb.org/portal/content/first-workshop-information-extraction-scientific-publications-wiesp-aacl-ijcnlp-2022): Full-Text Argumentation Mining on Scientific Publications.

Scholarly Argumentation Mining (SAM) has recently gained attention due to its potential to help scholars with the rapid growth of published scientific literature. It comprises two subtasks: argumentative discourse unit recognition (ADUR) and argumentative relation extraction (ARE), both of which are challenging since they require e.g. the integration of domain knowledge, the detection of implicit statements, and the disambiguation of argument structure. While previous work focused on dataset construction and baseline methods for specific document sections, such as abstract or results, full-text scholarly argumentation mining has seen little progress. In this work, we introduce a sequential pipeline model combining ADUR and ARE for full-text SAM, and provide a first analysis of the performance of pretrained language models (PLMs) on both subtasks. We establish a new SotA for ADUR on the Sci-Arg corpus, outperforming the previous best reported result by a large margin (+7% F1). We also present the first results for ARE, and thus for the full AM pipeline, on this benchmark dataset. 

## Setup

```
pip install -r requirements.txt
pip install -r requirements_analysis.txt
```

## Run
To run the experiments, you can follow the steps mentioned below. Note that scripts to reproduce the published 
results can be found in the [experiments](experiments) folder, especially [here](experiments/prediction) to
generate the predictions and [here](experiments/evaluation) to calculate the evaluation scores. 

### Training

**NOTE:** To train with cross validation refer to cross_validation [readme](training_scripts/cross_validation/README.md).

#### ADU Recognition

```bash
allennlp \
train \
-s experiments/training/adu/adu_best \
-f allennlp_configs/adu_best.jsonnet
```

#### Argumentative Relation Extraction

```bash
allennlp \
train \
-s experiments/training/rel/rel_best \
-f allennlp_configs/rel_best.jsonnet \
-o "{\"dataset_reader.add_negative_relations_portion\":-1.0}"
```
**NOTE** : To perform hyperparameter tuning follow the guide in hpt [readme](training_scripts/hpt/readme.md).

### Prediction

Note that scripts to reproduce the published results and their actual output can be found in 
[experiments/prediction](experiments/prediction).

#### ADU Recognition

1. Predicting ADUs and saving only GOLD ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"dataset_reader.show_gold\":true,\"dataset_reader.show_prediction\":false,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits\":{\"test\":\"30:\"}}" \
--use-dataset-reader \
--cuda-device 0 \
--output-file experiments/prediction/adu/goldonly \
--batch-size 8 \
--silent \
PATH/TO/ADU/MODEL \
./dataset_scripts/sciarg.json@test
```
Replace `PATH/TO/ADU/MODEL` with location where adu model is saved. For instance if you
run training command for ADU detection mentioned above then model will be saved in `experiments/training/adu/adu_best`
2. Predicting ADUs and saving only predicted ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"dataset_reader.show_gold\":false,\"dataset_reader.show_prediction\":true,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits\":{\"test\":\"30:\"}}" \
--use-dataset-reader \
--cuda-device 0 \
--output-file experiments/prediction/adu/predictiononly \
--batch-size 8 \
--silent  \
PATH/TO/ADU/MODEL \
./dataset_scripts/sciarg.json@test
```


#### Argumentative Relation Extraction
1. Predicting relations and saving only GOLD relations from GOLD ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":true,\"dataset_reader.show_prediction\":false,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\"30:\"}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@gold_adus/goldonly \
PATH/TO/REL/MODEL \
./dataset_scripts/sciarg.json@test
```
Replace `PATH/TO/REL/MODEL` with location where REL model is saved. For instance if you
run training command for relation extraction mentioned above then model will be saved in `experiments/training/rel/rel_best`
2. Predicting relations and saving only prediction relations from GOLD ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":false,\"dataset_reader.show_prediction\":true,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\"30:\"}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@gold_adus/predictiononly \
PATH/TO/REL/MODEL \
./dataset_scripts/sciarg.json@test
```

3. Predicting relations and saving GOLD and predicted relations from predicted ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":true,\"dataset_reader.show_prediction\":true,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\":\"}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@predicted_adus/gold_and_prediction \
PATH/TO/REL/MODEL \
PATH/TO/PREDICTED/ADUS/WITH/PREDICTION_ONLY@test
```
Replace `PATH/TO/PREDICTED/ADUS/WITH/PREDICTION_ONLY` with the location where 
predicted ADUS with only predictions are saved. For instance if you predict adus 
using command mentioned above then predicted ADUs with prediction only will be saved
at `experiments/prediction/adu/predictiononly`

4. Predicting relations and saving only prediction relations from predicted ADUs
```bash
allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":false,\"dataset_reader.show_prediction\":true,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\":\"}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@predicted_adus/predictiononly \
PATH/TO/REL/MODEL \
PATH/TO/PREDICTED/ADUS/WITH/PREDICTION_ONLY@test
```


### Evaluation

Note that scripts to reproduce the published results and their actual output can be found in 
[experiments/evaluation](experiments/evaluation).

1. Using AllenNLP `evaluate`
#### ADU Recognition
```bash
allennlp \
evaluate \
-o "{\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits\":{\"test\":\"30:\"}, \"model.calculate_weak_span_f1\":true}" \
--cuda-device 0 \
--batch-size 8 \
--output-file experiments/evaluation/using_allennlp/adu/metrics.json \
PATH/TO/ADU/MODEL \
./dataset_scripts/sciarg.json@test
```
You can find evaluation results [here](experiments/evaluation/using_allennlp/adu/best_uncased_10r5ge6a.json).


#### Argumentative Relation Extraction

```bash
allennlp \
evaluate \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null, \"dataset_reader.dataset_splits\":{\"test\":\"30:\"}}" \
--cuda-device 0 \
--batch-size 128 \
--output-file experiments/evaluation/using_allennlp/rel@gold_adus/metrics.json \
PATH/TO/REL/MODEL \
./dataset_scripts/sciarg.json@test
```



2. Using our evaluation pipeline ([calculate_metric.py](analysis/calculate_metric.py))

#### ADU Recognition

```bash
python analysis/calculate_metric.py \
--path_gold PATH/TO/PREDICTED/ADUS/WITH/GOLD_ONLY \
--path_predicted PATH/TO/PREDICTED/ADUS/WITH/PREDICTION_ONLY \
--out_dir experiments/evaluation/using_pipeline/adu/metrics
```
Replace `PATH/TO/PREDICTED/ADUS/WITH/GOLD_ONLY` with location where predicted adus
with only gold labels are saved. For instance, if you run prediction command mentioned
above it will be saved at `experiments/prediction/adu/goldonly`.

If you want to replicate metrics calculated from our best model (can be found [here](experiments/evaluation/using_pipeline/adu/best_uncased_10r5ge6a)) 
then replace `PATH/TO/PREDICTED/ADUS/WITH/GOLD_ONLY` with `experiments/prediction/adu/best_uncased_10r5ge6a_goldonly`
and `PATH/TO/PREDICTED/ADUS/WITH/PREDICTION_ONLY` with `experiments/prediction/adu/best_uncased_10r5ge6a_predictiononly`
#### Argumentative Relation Extraction

1. Evaluating relation extraction using GOLD ADUs

```bash
python analysis/calculate_metric.py \
--path_gold PATH/TO/PREDICTED/REL@GOLD_ADU/GOLD_ONLY \
--path_predicted PATH/TO/PREDICTED/REL@GOLD_ADU/PREDICTION_ONLY \
--out_dir experiments/evaluation/using_pipeline/rel@gold_adus/best_uncased_257eyrv1
```

In order to replicate metric calculated from our best model which can be found [here](experiments/evaluation/using_pipeline/rel@gold_adus/best_uncased_257eyrv1),
you can replace `PATH/TO/PREDICTED/REL@GOLD_ADU/GOLD_ONLY` with `experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly` 
and `PATH/TO/PREDICTED/REL@GOLD_ADU/PREDICTION_ONLY` with `experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_predictiononly`



2. Evaluating relation extraction using predicted ADUs
```bash
python analysis/calculate_metric.py \
--path_gold PATH/TO/PREDICTED/REL@GOLD_ADU/GOLD_ONLY \
--path_predicted PATH/TO/PREDICTED/REL@PREDICTED_ADU/PREDICTION_ONLY \
--out_dir experiments/evaluation/using_pipeline/rel@predicted_adus/best_uncased_257eyrv1
```

In order to replicate metric calculated from our best model which can be found [here](experiments/evaluation/using_pipeline/rel@predicted_adus/best_uncased_257eyrv1),
you can replace `PATH/TO/PREDICTED/REL@GOLD_ADU/GOLD_ONLY` with `experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly` 
and `PATH/TO/PREDICTED/REL@PREDICTED_ADU/PREDICTION_ONLY` with `experiments/prediction/rel@predicted_adus/best_uncased_257eyrv1_predictiononly`


