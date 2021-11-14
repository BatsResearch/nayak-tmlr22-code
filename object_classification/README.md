# Zero-shot object classification
The code was adapted from `https://github.com/cyvius96/DGP`.

## Setup
```
mkdir data
cd data
mkdir subgraphs
cd subgraphs
```

Download `ilsvrc_graph.zip`, `awa2_graph.zip`, `apy_graph.zip`, and `wordnet_graph.zip` from [google drive](https://drive.google.com/drive/u/0/folders/1uNXsC4l6hTZQp31edyPxRPcig3_7vhuy) and extract the zip in `data/subgraphs`.

If you want to run the baseline graph experiments (gcnz, sgcn, and dgp),  download `dense_graph.json` and `induced_graph.json` from [google drive](https://drive.google.com/drive/folders/1oNs1NNaMCClTO6el7qoXxPtwcdAXqLWa?usp=sharing) to `data`.

## Download datasets

- **ImageNet**: Download ImageNet from [http://image-net.org/download](http://image-net.org/download) and place them in `materials/datasets/imagenet`.

- **Animals with Attributes 2**: You can download the AWA2 dataset from [https://cvml.ist.ac.at/AwA2/AwA2-data.zip](https://cvml.ist.ac.at/AwA2/AwA2-data.zip).
This dataset is 13GB.
Extract the dataset from both the dataset and place them in `materials/datasets/awa2`.

- **Attribute Pascal and Attribute Yahoo (aPY)**:
Download a-yahoo dataset from [http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz](http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz) and place them in `materials/datasets/apy`
Download a-pascal dataset from [http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)

Extract the dataset from both the dataset and place them in `materials/datasets/apy`.

## Setting up AWA2 and aPY dataset
We follow the Proposed Split (PS) introduced in the Good, Bad, and Ugly paper.
To setup the training datasets for AWA2 and aPY datasets, we copy files
to a separete directory, possibly crop them, and rename the files with the
indices in the good, bad, and ugly splits.

For AWA2
```
cd materials
python gbu_awa2.py
```

For aPY
```
cd materials
python gbu_apy.py
```

## Train
We train the models with L2 loss between the pretrained ResNet model and the graph neural network features.
### ImageNet
```
python imagenet_train.py --label_encoder trgcn --max-epoch 3000 --resnet-50
```

You can run experiments for the following options `--label_encoder {gcn, gat, lstm, trgcn}`.

### AWA2/aPY
```
python train.py --label_encoder trgcn --max-epoch 1000 --seed 0
```

You can replace `--label_encoder` with `gcn`, `gat`, `rgcn`, `lstm`, or `trgcn` for ZSL-KG ablations.
For the baseline you can replace `--label_encoder` with `gcnz`, `sgcn`, and `dgp`.
We run all the experiments for 5 seed values {0, 1, 2, 3, 4}.

## Finetune
We freeze the `.pred` vectors and finetune the backbone ResNet model.
### ImageNet
```
python finetune_imagenet.py --pred <imagenet .pred path> --train-dir materials/dataset/imagenet
```
### AWA2
```
python finetune_awa2.py --pred <pred path> --train-dir materials/awa2/gbu --fold 0 --num-epochs 25
```
### aPY
```
python finetune_apy.py --pred <pred path> --train-dir materials/apy/gbu --fold 0 --num-epochs 25
```

## Evaluate

### ImageNet
Use `--consider-trains` to test generalized zero-shot learning, omit the flag for traditional zero-shot learning.
```
python evaluate_imagenet.py --pred <.pred file> --test-set <type> --resnet-50 --cnn <path to epoch 20> --consider-trains
```
`<type>` can be `2-hops`, `3-hops`, and `all`.

### AWA2

For generalized zero-shot learning.
```
python evaluate_awa2.py --pred <path to .pred> --gamma 3.0 --cnn <path to epoch-24.pth> --consider-trains --train-dir materials/datasets/gbu/awa2
```
For traditional zero-shot learning.
```
python evaluate_awa2.py --pred <path to .pred> --train-dir materials/datasets/gbu/awa2
```

### aPY

For generalized zero-shot learning.
```
python evaluate_apy.py --pred <path to .pred> --gamma 2.0 --cnn <path to epoch-24.pth> --consider-trains --train-dir materials/datasets/gbu/apy
```
For traditional zero-shot learning.
```
python evaluate_apy.py --pred <path to .pred> --train-dir materials/datasets/gbu/apy
```

## Resource
All the experiments are trained and evaluated on Nvidia Tesla V100 with 32GB.
