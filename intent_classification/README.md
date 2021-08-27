# Zero-shot Intent Classification

## Basic Setup
```
mkdir data
cd data
```

- Download `snips_graph.zip` from [google drive](https://drive.google.com/drive/u/0/folders/1uNXsC4l6hTZQp31edyPxRPcig3_7vhuy) and extract the zip in `data/subgraphs`.

- If you want to run the baseline graph experiments (gcnz, sgcn, and dgp),  download `dense_graph.json` and `induced_graph.json` from [google drive](https://drive.google.com/drive/folders/108MO3qjtRfcsdCMl_bQLo7LOoT0-i0XJ?usp=sharing) to `data`.


## Dataset
Download the dataset `snips` from [google drive](https://drive.google.com/drive/folders/1ZiVEmhs1vt5VGH1ZB9eecLELiJEYzO0C?usp=sharing) and place them in `data/snips`.

## Running experiments
```
python train.py --dataset bbn --label_encoder_type trgcn --seed 0
```

<!-- You can replace dataset with `bbn` and `ontonotes`.
You can also replace `--label_encoder_type` with the following - `trgcn`, `lstm`, `gcn`, `gat`, and `rgcn`. -->
You can vary the arguments to get benchmark different class encoders:
```
--dataset {bbn, ontonotes}
--label_encoder_type {gcn, gat, rgcn, lstm, trgcn, sgcn, gcnz, dgp}
```
WordNet baselines: `sgcn, gcnz, dgp`

ZSL-KG: `gcn, gat, rgcn, lstm, trgcn`