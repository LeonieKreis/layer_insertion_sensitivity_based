# Generate a dataset for the pytorch frame work for a one-dimensional regression problem 
# where the underlying function is a piecewise linear and continuous function.

import numpy as np
import torch
import random
import json
import matplotlib.pyplot as plt

class StepFunctionDataset(torch.utils.data.Dataset):
    def __init__(self, n_points=1000, n_steps=10, noise=0.1, seed=0):
        self.n_points = n_points
        self.n_steps = n_steps
        self.noise = noise
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.x = np.linspace(0, 1, n_points)
        self.y = self._generate_data()
        
    def _generate_data(self):
        y = np.zeros(self.n_points)
        step_size = 1 / self.n_steps
        intercept = 0
        if self.n_steps == 5:
            linear_scales = [2,0.5,-1,4,2]
        else:
            linear_scales = np.random.uniform(-5, 5., self.n_steps)
        
        for i in range(self.n_steps):
            start = i * step_size # beginning of linear part
            end = (i + 1) * step_size # end of linear part
            mask = (self.x >= start) & (self.x <= end)
            # Calculate the slope for the linear segment
            slope = linear_scales[i] / step_size
            # Calculate the linear values for the segment
            y[mask] = slope * (self.x[mask] - start) + intercept
            intercept = y[mask][-1]
        y += self.random.normal(0, self.noise, self.n_points)
        return y
    
    #def _generate_data(self):
    #    y = np.zeros(self.n_points)
    #    step_size = 1 / self.n_steps
    #    for i in range(self.n_steps):
    #        start = i * step_size
    #        end = (i + 1) * step_size
    #        y += np.where((self.x >= start) & (self.x < end), i, 0)
    #    y += self.random.normal(0, self.noise, self.n_points)
    #    return y
    
    def __len__(self):
        return self.n_points
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return torch.tensor(x).float().unsqueeze(0), torch.tensor(y).float().unsqueeze(0)
    
    def save(self, filename):
        data = {
            'n_points': self.n_points,
            'n_steps': self.n_steps,
            'noise': self.noise,
            'seed': self.seed,
            'x': self.x.tolist(),
            'y': self.y.tolist()
        }
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        dataset = cls()
        dataset.n_points = data['n_points']
        dataset.n_steps = data['n_steps']
        dataset.noise = data['noise']
        dataset.seed = data['seed']
        dataset.x = np.array(data['x'])
        dataset.y = np.array(data['y'])
        return dataset
    
    def plot(self):
        plt.plot(self.x, self.y, 'o')
        plt.show()


def gen_steps_dataset(n_points=1000, n_steps=10, noise=0.1, seed=0, batchsize=20):
    dataset = StepFunctionDataset(n_points, n_steps, noise, seed)

    params = {'batch_size': batchsize,  # specify minibatchsize!
            'shuffle': True,
            'num_workers': 0}

    params2 = {'batch_size': int(batchsize/3),
            'shuffle': True,
            'num_workers': 0}
    
    # split the dataset into training and validation after shuffling
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(np.floor(0.75 * n))
    train_indices, val_indices = indices[:split], indices[split:]
    dataset_train = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset, val_indices)


    training_generator = torch.utils.data.DataLoader(dataset_train, **params)
    validation_generator = torch.utils.data.DataLoader(dataset_val, **params2)

    data_X = dataset.x
    data_y = dataset.y

    return training_generator, validation_generator, data_X, data_y

def plot_predictor_withdataset(data_X, data_y, model, device, title='performance of predictor', save=False, only_data=True):
    plt.plot(data_X, data_y, 'o', label='data')
    if only_data:
        plt.plot(data_X, model(torch.tensor(data_X).float().to(device).unsqueeze(1)).detach().cpu().numpy(), label='predictor')
    else:
        x = np.linspace(-10, 10, 1000)
        plt.plot(x, model(torch.tensor(x).float().to(device).unsqueeze(1)).detach().cpu().numpy(), label='predictor')
    plt.title(title)
    plt.show()
    if save:
        plt.savefig(f'figs/{title}.pdf')

if __name__ == '__main__':
    dataset = StepFunctionDataset(n_steps=5,noise=0)
    dataset.plot()
    #dataset.save('step_function.json')
    #dataset2 = StepFunctionDataset.load('step_function.json')
    #dataset2.plot()
    
    # Example of how to use the dataset with a DataLoader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    #for x, y in dataloader:
       #print(x, y)
    #    break