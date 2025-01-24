# Bidirectional Action Decoding

This repo contains the official implementation of Bidirectional Action Decoding.

> Bidirectional Decoding (BiD) samples multiple action chunks at each time step and searches for the optimal action based on two criteria:
>   1. backward coherence, which favors actions close to the decision made in the previous time step
>   2. forward contrast, which favors actions close to near-optimal long-horizon plans and far from sub-optimal short-horizon ones
> 
> By coupling decisions within and across action chunks, our method promotes strong temporal dependencies over multiple time steps while maintaining high reactivity to unexpected environment dynamcis.

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
cd -rf /iris/u/yuejliu/research/bid/ckpt <your_path>/bid/ckpt
```

### Acknowledgement

Code adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
