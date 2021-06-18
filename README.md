# Solution of SIGIR Ecom Data Challenge 2021

## Overview

Coveo hosts the 2021 [SIGIR eCom](https://sigir-ecom.github.io/data-task.html) Data Challenge

## Requirements

- torch==1.8.1
- tqdm==4.60.0
- numpy==1.20.2

- boto3==1.15.8

- python-dotenv==0.13.0

## Getting Started

```shell
mkdir saved log results
mkdir dataset & cd dataset
mkdir new prepared raw
```

The path of raw dataset is `./dataset/raw`

### Pre-Process

Run the scripts in `./scripts`

### Models

#### txt embedding

```shell
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit txt
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit txt --evaluate
```

#### deepwalk embedding(url-sess-item)

```shell
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit dw
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit dw --evaluate
```

#### deepwalk embedding(item-item)

```shell
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit dw_i-i
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit dw_i-i --evaluate
```

#### rand embedding

```shell
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit rand
 python train.py --model GRU4Rec --device 0 --lr 1e-4 --seq_mode sku --commit rand --evaluate
```

### Post-Process

Run `ensemble.ipynb`
