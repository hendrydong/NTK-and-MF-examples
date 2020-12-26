# Repopulation 

## Repopulated feature vs Random feature

### 1. Feature space

Fix the feature space, we only optimize the second layer of NN, equivalent with logistic regression.

Repopulated
```
python main_loadvae_cuda_tanh.py
```

Random 
```
python main_cuda_random.py
```

### 2. Tangent space

Fix the feature space, we optimize the second layer of NN and tangent space of feature. 

Repopulated
```
python main_cuda_ntk_load.py
```

Random 
```
python main_cuda_ntk.py
```


## Distill from repopulated feature samples (importance sampling vs uniform)

Pretrain large model (Not required, we have provided a pretrained one)
```
python main_mnist_10000.py
```

Importance 
```
python main_mnist_is.py
```

Uniform 
```
python main_mnist_us.py
```


