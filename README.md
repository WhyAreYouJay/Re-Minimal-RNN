# Re-minimal RNN: Max-return SM with minimal RNNs

This repository combines the parallelizable and sleeker minimal RNN from [Were RNNs all we needed](https://arxiv.org/abs/2410.01201), with the maximum return prediction model of the [Reinformer](https://arxiv.org/abs/2405.08740).

#### Running locally
The needed libraries can be installed with **pip install -r requirements.txt**

You can change the settings with command line arguments, look into available arguments in the file **single_full_seed_run.py**.

Consult below for the settings to reproduce our scores for the D4RL Hopper,Walker2D and HalfCheetah environment benchmarks.

#### Running on Colab/AWS and other services
- Upload the colab to the GPU providers server.
- Simply run all cells, it will install and run everything for you.
- You can change the last cells command line arguments to the hyperparameter settings you like


## Settings
###### Attach these command line arguments to run **single_full_seed_run.py** this will run all 3 seeds in succesion
###### Setting --stacked argument to any value sets the boolean to true. To run without stacked, leave the default setting
###### **single_seed_run.py** allows to run a singular seed. We have run with the seeds (0,42,2024)
###### Minimal LSTM can be run with the last argument (--block_type "minlstm"). Minimal GRU is the default
**HalfCheetah-Medium** :  --env halfcheetah --dataset medium --embed_dim 128 --batch_size 128 --K 5 --lr 0.0001 --n_layers 3 --wd 0.0001 --tau 0.9 --warmup_steps 10000 --stacked True --expansion_factor 2.0

**HalfCheetah-Medium-Expert** : --env halfcheetah --dataset medium_expert --embed_dim 128 --batch_size 128 --K 20 --lr 0.0001 --n_layers 3 --wd 0.0001 --tau 0.9 --warmup_steps 10000 --stacked True --expansion_factor 2.0

**Walker2D-Medium** : --env walker2d --dataset medium --embed_dim 128 --batch_size 256 --K 5 --lr 0.0001 --n_layers 3 --wd 0.0001 --tau 0.9 --warmup_steps 10000 --expansion_factor 2.0

**Walker2D-Medium-Expert** : --env walker2d --dataset medium_expert --embed_dim 128 --batch_size 256 --K 20 --lr 0.00015 --n_layers 3 --wd 0.0001 --tau 0.99 --warmup_steps 10000 --stacked True --expansion_factor 2.0 

**Hopper-Medium** : --env hopper --dataset medium --embed_dim 256 --batch_size 128 --K 5 --lr 0.0001 --n_layers 3 --wd 0.0001 --tau 0.999 --warmup_steps 10000 --stacked True --expansion_factor 1.0

**Hopper-Medium-Expert** : --env hopper --dataset medium_expert --embed_dim 128 --batch_size 128 --K 20 --lr 0.001 --n_layers 3 --wd 0.0001 --tau 0.9 --warmup_steps 10000 --stacked True --expansion_factor 1.5

# Double DQN expert for Connect-four
#### Running locally
To run locally simply run the train.py file located in experts/DQN. By default the trained agent will be evaluated against a random action policy every 500 episodes where the agent will play 1000 games and return a win rate, draw rate and loss rate.