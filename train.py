'''
Jihyun Lim, M.S. <wlguslim@inha.edu>
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
'''

import numpy as np
import tensorflow as tf
import client_sampling
from mpi4py import MPI
from tqdm import tqdm

class framework:
    def __init__ (self, models, dataset, solver, **kargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.models = models
        self.dataset = dataset
        self.solver = solver
        self.num_epochs = kargs["num_epochs"]
        self.lr = kargs["lr"]
        self.decay_epochs = kargs["decay_epochs"]
        self.num_classes = kargs["num_classes"]
        self.num_workers = kargs["num_workers"]
        self.num_clients = kargs["num_clients"]
        self.num_candidates = kargs["num_candidates"]
        self.average_interval = kargs["average_interval"]
        self.do_checkpoint = kargs["do_checkpoint"]
        self.num_local_workers = int(self.num_workers / self.size)
        self.lr_decay_factor = 10
        self.participations = np.zeros((self.num_clients))
        if self.num_classes == 1:
            self.valid_acc = tf.keras.metrics.BinaryAccuracy()
        else:
            self.valid_acc = tf.keras.metrics.Accuracy()
        self.checkpoint = tf.train.Checkpoint(models = models, optimizers = self.solver.local_optimizers)
        for optimizer in self.checkpoint.optimizers:
            optimizer.lr.assign(self.lr)
        checkpoint_dir = "./checkpoint"
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                              directory = checkpoint_dir,
                                                              max_to_keep = 3)
        self.sampler = client_sampling.sampling(self.num_clients, self.num_workers, self.num_candidates,self.checkpoint)
        
        # Create all the local datasets first.
        self.train_datasets = []
        for i in range (self.num_clients):
            self.train_datasets.append(self.dataset.train_dataset(i))

        # Resume if any checkpoints are in the current directory.
        self.checkpoint.models[0].summary()
        self.resume()

    def resume (self):
        self.epoch_id = 0
        if self.checkpoint_manager.latest_checkpoint:
            self.epoch_id = int(self.checkpoint_manager.latest_checkpoint.split(sep='ckpt-')[-1]) - 1
            if self.rank == 0:
                print ("Resuming the training from epoch %3d\n" % (self.epoch_id))
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def train (self):
        
        # Broadcast the parameters from rank 0 at the first epoch.
        start_epoch = self.epoch_id
        if start_epoch == 0:
            self.broadcast_model()

        for epoch_id in range (start_epoch, self.num_epochs):
            
            # LR decay
            if epoch_id in self.decay_epochs:
                self.participations = np.zeros((self.num_clients))
                lr_decay = 1 / self.lr_decay_factor
                for optimizer in self.checkpoint.optimizers:
                    optimizer.lr.assign(optimizer.lr * lr_decay)
                
            # Client selection.
            active_devices = self.sampler.multi_metric()   
            #active_devices = self.sampler.random()
            offset = self.num_local_workers * self.rank
            local_devices = active_devices[offset: offset + self.num_local_workers]

            # Collect the active train datasets.
            train_datasets = []
            for i in range (len(local_devices)):
                train_datasets.append(self.train_datasets[local_devices[i]])

            # Training loop.
            local_losses = []
            local_norms = []
            for local_id in tqdm(range(self.num_local_workers), ascii=True):
                local_loss, local_norm = self.solver.round(epoch_id, self.checkpoint.models, train_datasets, local_id)
                local_losses.append(local_loss.result().numpy())
                local_norms.append(local_norm.result().numpy())

            # Update the local information.
            aggregated_losses = np.array(self.comm.allgather(local_losses)).flatten()
            self.sampler.update_loss(aggregated_losses)

            aggregated_norms = np.array(self.comm.allgather(local_norms)).flatten()
            self.sampler.update_norm(aggregated_norms)
                
            # Average the local updates and apply it to the global model.
            self.solver.average_model(self.checkpoint, epoch_id)

            # Collect the global training results (loss and accuracy).
            global_loss = self.comm.allreduce(sum(local_losses), op = MPI.SUM) / self.num_workers

            # Collect the global validation accuracy.
            local_acc = self.evaluate()
            global_acc = self.comm.allreduce(local_acc, op = MPI.MAX)

            # Checkpointing
            if self.do_checkpoint == True and epoch_id < 250:
                self.checkpoint_manager.save()

            # Logging.
            if self.rank == 0:
                print ("Epoch " + str(epoch_id) +
                       " lr: " + str(self.checkpoint.optimizers[0].lr.numpy()) +
                       " validation acc = " + str(global_acc) +
                       " training loss = " + str(global_loss))
                f = open("acc.txt", "a")
                f.write(str(global_acc) + "\n")
                f.close()
                f = open("loss.txt", "a")
                f.write(str(global_loss) + "\n")
                f.close()

    def evaluate (self):
        valid_dataset = self.dataset.valid_dataset()
        self.valid_acc.reset_states()
        for i in tqdm(range(self.dataset.num_valid_batches), ascii=True):
            data, label = valid_dataset.next()
            predicts = self.checkpoint.models[0](data)
            if label.shape[-1] == 1:
                self.valid_acc(label, predicts)
            else:
                self.valid_acc(tf.argmax(label, 1), tf.argmax(predicts, 1))
        accuracy = self.valid_acc.result().numpy()
        return accuracy

    def broadcast_model (self):
        for i in range (len(self.checkpoint.models[0].trainable_variables)):
            param = self.checkpoint.models[0].trainable_variables[i]
            param = self.comm.bcast(param, root=0)
            for j in range (self.num_local_workers):
                self.checkpoint.models[j].trainable_variables[i].assign(param)
