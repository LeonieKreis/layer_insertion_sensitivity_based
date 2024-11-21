from heatmap import make_movie
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

repopathgrads = os.path.join(base_dir, 'heatmaps/grads_1732114385.9946206/')
repopathvals = os.path.join(base_dir, 'heatmaps/vals_1732114385.9946973/')

# make movie out of the heatmaps for each weight (=num) separate
num=0
no_params = 3
no_epochs = 500
no_batches = 8

for num in range(no_params):
    make_movie(repopathgrads, num, no_epochs=no_epochs, no_batches=no_batches)
    make_movie(repopathvals, num, no_epochs=no_epochs, no_batches=no_batches)



repopathgrads2 = 'heatmaps/grads_1732116723.3240142/'
repopathvals2 = 'heatmaps/vals_1732116723.3604677/'
no_epochs2 = 500
no_batches2 = 8

no_params2 = no_params +2
for num in range(no_params2):
    make_movie(repopathgrads2, num, no_epochs=no_epochs2, no_batches=no_batches2)
    make_movie(repopathvals2, num, no_epochs=no_epochs2, no_batches=no_batches2)
   