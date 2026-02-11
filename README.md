# Multi-Metric Client Activation Method for Fast and Accurate Federated Learning
This is an official implementation of the following paper:
> Jihyun Lim, Tuo Zhang and Sunwoo Lee.<br>
 **Multi-Metric Client Activation Method for Fast and Accurate Federated Learning**  
_ACM Transactions on Intelligent Systems and Technology_.
>

The paper link is here ➔  [Paper](https://dl.acm.org/doi/abs/10.1145/3793668) <br>

## Software Requirements
 * tensorflow2 (<= 2.15.1)
 * tensorflow_datasets
 * python3
 * mpi4py
 * tqdm

## Instructions
### Training
 1. Set hyper-parameters properly in `config.py`.
 2. Run training.
    ```
    mpirun -np 2 python3 main.py
    ```
### Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 1. `loss.txt`: An output file that contains the training loss for every epoch.
 2. `acc.txt`: An output file that contains the validation accuracy for every epoch.
 3. `./checkpoint`: The checkpoint files generated after every epoch. This directory is created only when `checkpoint` is set to 1 in `config.py`.


## Supported Federated Learning Features
 * FedAvg
 

## Supported Datasets
 * CIFAR-10

## Questions / Comments
 * Jihyun Lim(wlguslim@inha.edu) <a href="https://github.com/jjihyunh"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="24"/></a>
 * Sunwoo Lee (sunwool@inha.ac.kr) <a href="https://github.com/swblaster"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="24"/></a>
