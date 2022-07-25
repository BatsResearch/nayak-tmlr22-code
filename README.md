

# ZSL-KG Experiments
ZSL-KG is a general-purpose zero-shot learning framework with a novel transformer graph convolutional network (TrGCN) to learn class representation from common sense knowledge graphs.

This is the codebase for all the experiments mentioned in [Zero-shot Learning with Common Sense Knowledge graphs](https://arxiv.org/abs/2006.10713).

## Code Organization

The code is organized by task, namely:

1. `intent_classification`
2. `fine_grained_entity_typing`
3. `object_classification`
4. `graph_utils`

Refer to the individual directories (`intent_classification`, `fine_grained_entity_typing`, and `object_classification`) to run the zero-shot experiments.

### Querying the ConceptNetDB
In the `graph_utils` directory, we include example code to query and preprocess the 2-hop neighbourhood for the ImageNet classes from the conceptnetdb.
While the preprocessing the graph is relatively simple, setting up the initial ConceptNet database with the [official guide](https://github.com/commonsense/conceptnet5/wiki/Build-process) could be time-consuming.
To easily reproduce our experiments, we're releasing the all the knowledge graph-related data for the experiments on [google drive](https://drive.google.com/drive/folders/1nPO5QWdbqYzro7P9dxwm5zhA01-QSoh5?usp=sharing).


## Setup
```
conda create --name zsl_kg python=3.7
conda activate zsl_kg
pip install -r zsl-kg-requirements.txt
```
You can go to the respective directory and run the experiments.

## Data
We include all the knowledge graph-related data for the experiments on [google drive](https://drive.google.com/drive/folders/1nPO5QWdbqYzro7P9dxwm5zhA01-QSoh5?usp=sharing).

## Citation
Please cite the following [paper](https://arxiv.org/abs/2006.10713) if you are using our framework.

```
@article{nayak:tmlr22,
  Author = {Nayak, N. V. and Bach, S. H.},
  Title = {Zero-Shot Learning with Common Sense Knowledge Graphs},
  Journal = {Transactions on Machine Learning Research (TMLR)},
  Year = {2022}}
```
