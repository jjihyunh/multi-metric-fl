'''
Jihyun Lim, M.S. <wlguslim@inha.edu>
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
'''

import tensorflow as tf
import config as cfg
from train import framework
from mpi4py import MPI
from fedavg import FedAvg
from model import ResNet20
from feeder_cifar import cifar
tf.keras.utils.disable_interactive_logging()         
       
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    local_rank = rank % len(gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')

    if rank == 0:
        print ("---------------------------------------------------")
        print ("dataset: " + "CIFAR-10")
        print ("number of workers: " + str(cfg.num_workers))
        print ("average interval: " + str(cfg.average_interval))
        print ("batch_size: " + str(cfg.batch_size))
        print ("training epochs: " + str(cfg.num_epochs))
        print ("---------------------------------------------------")

    num_clients = int(cfg.num_workers / cfg.active_ratio)
    
    # Dataset
    dataset = cifar(batch_size = cfg.batch_size,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_classes = cfg.num_classes,
                        alpha = cfg.alpha)
    
    # Models
    num_local_workers = cfg.num_workers // size
    models = []
    for i in range (num_local_workers):
        models.append(ResNet20(cfg.weight_decay, cfg.num_classes).build_model())
    
    # FedAvg
    solver = FedAvg(num_classes = cfg.num_classes,
                        num_workers = cfg.num_workers,
                        average_interval = cfg.average_interval)
    
    # Training
    trainer = framework(models = models,
                        dataset = dataset,
                        solver = solver,
                        num_epochs = cfg.num_epochs,
                        lr = cfg.lr,
                        decay_epochs = cfg.decay_epochs,
                        num_classes = cfg.num_classes,
                        num_workers = cfg.num_workers,
                        num_clients = num_clients,
                        num_candidates = cfg.num_candidates,
                        average_interval = cfg.average_interval,
                        do_checkpoint = cfg.checkpoint)
    trainer.train()
