'''
Jihyun Lim, M.S. <wlguslim@inha.edu>
Sunwoo Lee, Ph.D. <sunwool@inha.ac.kr>
'''

batch_size = 32
lr = 0.2
num_classes = 10
num_epochs = 250
decay_epochs = {125, 180}
weight_decay = 0.0001
average_interval = 20
num_workers = 32 
num_candidates = 48
checkpoint = 0
quantizer_level = 10
mu = 0.001
'''
Federated Learning settings
1. Device activation ratio (0.25, 0.5, 1)
2. Dirichlet's concentration parameter (0.1, 0.5, 1)
'''
active_ratio = 0.25
alpha = 0.1
