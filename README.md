# AIMeetsBeautyChallenge
This is the source code of [Perfect Half Million Beauty Product Image Recognition Challenge](https://challenge2019.perfectcorp.com/index.html).

## Installation
* Env: python3.6
* Package:
```
pytorch 1.1
h5py
faiss
pandas
tensorboardX
tqdm
```

## Get started
1. train.py
```bash
python train.py --model-name resnet152 --dataset /data/dataset/AIMeetsBeauty --epochs 30 --batch-size 64 --lr 0.0001
```

2. get_feature.py
```bash
python get_feature.py --model-name resnet152 --dataset /data/dataset/AIMeetsBeauty --model-ckpt ./ckpts/resnet152.pkl --output-feature-path ./feature/resnet152.hdf5
```

3. predict.py
```bash
python predict.py --model-name resnet152 --model-ckpt ./ckpts/resnet152.pkl --feature-path ./feature/resnet152.hdf5 --test-path ./val_2019 --output-result-path ./result/resnet152.csv
```

4. evaluation.py
```bash
python evaluation.py --prediction-path ./result/predictions.csv --label-path ./val_2019.csv
```

5. ensemble_predict.py

Options: Using multiple models to perform ensemble predictions, which need to change the loading path of model checkpoint file and feature hdf5 file.
```bash
python ensemble_predict.py --prediction-path ./result/predictions_att.csv --label-path ./val_2019.csv
```
