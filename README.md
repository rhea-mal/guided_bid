# Guided Bidirectional Decoding

> Bidirectional Decoding (BiD) samples multiple action chunks at each time step and searches for the optimal action based on two criteria:
>   1. backward coherence, which favors actions close to the decision made in the previous time step
>   2. forward contrast, which favors actions close to near-optimal long-horizon plans and far from sub-optimal short-horizon ones
Here we implement guided diffusion for BID sampling to enhance efficiency, reduce latency required to achieve BID performance.
### Setup

Install dependencies of the diffusion policy (~20 min)
```
mamba env create -f conda_environment.yaml
mamba activate bid
```

Install additional dependencies
```
pip install -r requirement.txt
```

Download [online data](https://diffusion-policy.cs.columbia.edu/data/training/)
```
bash script/download_dataset.sh
```

### Decoding Scripts

The [sampler](diffusion_policy/sampler) folder contains a collection of test-time decoding / sampling algorithms.

- random sampling baseline
```
bash script/eval_random.sh
```

- backward coherence (ours)
```
bash script/eval_coherence.sh
```

- forward contrast (ours)
```
bash script/eval_contrast.sh
```

- bidirectional decoding (ours)
```
bash script/eval_bid.sh
```

- sbatch script
```
sbatch script/sbatch_eval.sh
```

### Training Scripts

- training from scratch
```
sbatch script/sbatch_train.sh
```

- pre-trained checkpoints
```
cd -rf /iris/u/rheamal/research/bid/ckpt <your_path>/bid/ckpt
```

### Acknowledgement

Code adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
