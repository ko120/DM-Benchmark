## Setup

Clone the respository

```
git clone https://github.com/ko120/DM-Benchmark.git
```

Create the conda environment

```
conda create -n "dm" python=3.10.8 ipython
conda activate dm
```

Install packages

``` 
pip install -r requirements.txt
```

## Usage

### Classification

All experimental settings can be found under `configs/experiments/classification_*`. The available classification datasets are `adult`
To train a model using our mixed loss (`MMD + NLL`), run

```
python train.py -m experiment=classification_mixed
```
