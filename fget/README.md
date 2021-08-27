## Fine-grained Entity Typing

### Setup
```
mkdir data
cd data
```

- Download `ontonotes_graph.zip` and `bbn_graph.zip` from [google drive](https://drive.google.com/drive/u/0/folders/1uNXsC4l6hTZQp31edyPxRPcig3_7vhuy) and extract the zip in `data/subgraphs`.

- If you want to run the baseline graph experiments (gcnz, sgcn, and dgp),  download `dense_graph.json` and `induced_graph.json` from [google drive](https://drive.google.com/drive/folders/1UyOYLbw9YwKEbAbEqaoy3nIpprLQ_YtV?usp=sharing) to `data`.

## Dataset
Download the dataset `ontonotes` and `bbn` from [google drive](https://drive.google.com/drive/folders/1zIoS1dJgkksKoEdYOXayBWL_lhJx7dpq?usp=sharing) and place them in `data/ontonotes` and `data/bbn`.

### Running ZSL-KG experiments
```
python train.py --dataset bbn --label_encoder_type trgcn --seed 0
```

You can vary the arguments to get benchmark different class encoders:
```
--dataset {bbn, ontonotes}
--label_encoder_type {otyper, dzet, gcn, gat, rgcn, lstm, trgcn, sgcn, gcnz, dgp}
```

ZSL-KG: `gcn, gat, rgcn, lstm, trgcn`

Existing baselines: `otyper, dzet`

WordNet baselines: `sgcn, gcnz, dgp`
