'''
Jihyun Lim, M.S. <wlguslim@inha.edu>
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
'''

from mpi4py import MPI
import numpy as np
import time

class sampling:
    def __init__ (self, num_clients, num_workers, num_candidates,checkpoint):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = num_workers
        self.num_clients = num_clients
        self.num_candidates = num_candidates
        self.num_local_workers = int(num_workers / self.size)
        self.num_local_clients = int(num_clients / self.size)
        self.local_losses = np.full((self.num_clients), np.Inf)
        self.fixed_losses = np.full((self.num_clients), np.Inf)
        self.local_norms = np.zeros((self.num_clients))
        self.active_devices = np.zeros((self.num_clients))
        self.rng = np.random.default_rng(int(time.time()))
        np.random.seed(int(time.time()))
        
    def random (self):
        self.active_devices = np.random.choice(np.arange(self.num_clients), size = self.num_workers, replace = False)
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)
        return self.active_devices

    def multi_metric (self):
        r = 256
        b = 8
        weights = np.ones((self.num_clients))
        lossprob = np.sort(np.argsort(self.local_losses)[-(self.num_workers + b):])
        normprob = np.sort(lossprob[np.argsort(self.local_norms[lossprob])[:(self.num_workers)]])
        weights[lossprob] *= r
        weights[normprob] *= r
        weights /= sum(weights)
        self.active_devices = self.rng.choice(np.arange(self.num_clients), size = self.num_workers, replace = False, p = weights)
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)
        return self.active_devices

    def update_loss (self, aggregated_losses):
        for i in range (len(aggregated_losses)):
            client_id = self.active_devices[i]
            self.local_losses[client_id] = aggregated_losses[i]

    def update_norm (self, aggregated_norms):
        for i in range (len(aggregated_norms)):
            client_id = self.active_devices[i]
            self.local_norms[client_id] = aggregated_norms[i]

    
