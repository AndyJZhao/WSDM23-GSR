# WSDM23-GSR
An implementation of the GSR model proposed in Self-Supervised Graph Structure Refinement for Graph Neural Network in WSDM23.
Please kindly note that this work is inspired by the great work [GAuG](https://arxiv.org/abs/2006.06830), especially the adding and removing edges part.

## Requirements

- Python >=3.8.5
- Pytorch == 1.8.1
- DGL == 0.6.2
- SciPy == 1.6.2
- NetworkX==2.5.1

## Usage

- Step 1 : Unzip the data.zip file
- Step 2: Run the experiments using the command below

to recover our results, specify "-l" option

```
python root_to_src_dir/src/models/GSR/trainGSR.py -dcora -l
```



## Usage

### Experiment Settings

- `dataset`: The dataset to run on.
- `gpu`: GPU id to use.
- `train_percentage`:The train ratio, 0 for default split.
- `load_default_config`: Whether load default config or use parsed config.
- `early_stop`: Number of epoch for early stop.
- `seed`: Training seed.

### Model Settings

- `epochs`: The maximum epoch to train in the fine-tuning process.
- `intra_weight`: The weight of intra contrastive loss, $\alpha$ in the paper.
- `fsim_weight`: The weight of feature similarity in estimating edge probability, $\beta$ in the paper.
- `add_ratio`: Percentage of non-existing edge to add in the graph refinement process.
- `rm_ratio`: Percentage of existing edge to remove in the graph refinement process.
- `fan_out`: Number of neighbors to sample in first and second order subgraphs.
- `p_epochs`: Number of pre-training epochs.
- `p_batch_size`: The pre-training batch size.
- `prt_lr`: The pre-training learning rate.

## Citation
If you find our work useful, please consider citing our work:
```
@inproceeding{zhao2023gsr,
  title={Self-Supervised Graph Structure Refinement for Graph Neural Networks},
  author = {Jianan Zhao and Qianlong Wen and Mingxuan Ju and Chuxu Zhang and Yanfang Ye},
  booktitle = {WSDM},
  year      = {2023},
}
```

