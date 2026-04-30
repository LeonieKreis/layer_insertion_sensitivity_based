#  ########## necessary imports ###############################################
import torch
import random
import os
import copy
import sys
import numpy as np
from torch import nn

sys.path.append('../layer_insertion_sensitivity_based')

from layer_insertion_loop import layer_insertion_loop
from train_and_test_ import train, check_testerror
from nets import feed_forward, two_weight_resnet, one_weight_resnet
from save_to_json import write_losses
from spirals_data_new import gen_spiral_dataset, plot_spirals

# ################# fix hyperparameters ###################################

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved



# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)

# Define hyperparameters

hidden_layers_start = 1
fix_width = [5]
no_iters = 1
lr_decrease_after_li = 1.
epochs = [350,1500]  # [10, 5, 5]
wanted_testerror = 0.
_type = 'fwd'
act_fun = nn.ReLU
interval_testerror = 1

batchsize = 450 # fullbatch
no_per_class = 300
r0=0.5
circles = 1


td, vd, data_X, data_y = gen_spiral_dataset(batchsize,no_per_class,r0,circles)

save =None# 'spirals.pdf'
plot_spirals(data_X, data_y, save=save)


