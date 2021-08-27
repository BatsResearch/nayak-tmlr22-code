

# ZSL-KG Experiments
ZSL-KG is a general-purpose zero-shot learning framework with a novel transformer graph convolutional network (TrGCN) to learn class representation from common sense knowledge graphs.

This is the codebase for all the experiments mentioned in [Zero-shot Learning with Common Sense Knowledge graphs](https://arxiv.org/abs/2006.10713).

The code is organized by task, namely:

1. intent_classification
2. fine_grained_entity_typing
3. object_classification

Refer to the individual directories to run the experiments.

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
@article{nayak:arxiv20,
  Author = {Nayak, Nihal V. and Bach, Stephen H.},
  Title = {Zero-Shot Learning with Common Sense Knowledge Graphs},
  Volume = {arXiv:2006.10713 [cs.LG]},
  Year = {2020}}
```