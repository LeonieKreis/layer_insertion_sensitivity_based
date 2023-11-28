from torch.utils import data
from random import shuffle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(0)
N = 400 # number of points per class
D = 2  # dimensionality
K = 2  # number of classes


number = N
r0 = 0.2
circles = 2


def spiral_xy(i,spiral_num, number, circles = 1, r0 = 0 ):
    '''
    '''
    phi = torch.tensor(0.) # start value of phi
    delta_phi = torch.tensor(2 * np.pi * circles / number) # increment which is added to phi for each new datapoint
    phi= phi+i*delta_phi
    
    r = r0 # initial radius
    delta_r = circles / number
    r = r +i* delta_r

    x= r*spiral_num*torch.cos(phi)
    y = r*spiral_num*torch.sin(phi)
    return [x,y]




def spiral(spiral_num):
    return [spiral_xy(i, spiral_num,number=N, circles=2, r0=0.1) for i in range(N)]


X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')
y[0:N] = np.zeros(N)  # represents -1
y[N:] = np.ones(N)



X[0:N] = spiral(-1)
X[N:] = spiral(1)


data_X = X
data_y = y

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    print("Running on : ", torch.cuda.device_count(), " GPUs!"
          if torch.cuda.device_count() > 1 else " GPU!")
else:
    print("Running on CPU!")


# create the data directory
print('Current working directory: ', os.getcwd())
if os.path.exists('data'):
    import shutil
    shutil.rmtree('data')
    print('Removed previously created data directory & contents')

os.mkdir('data')
print('New data directory created')

print(os.listdir())

# Join class labels vector onto features matrix ready for storage
Xy = np.block([[X, np.reshape(y, (len(y), 1))]])

# Method to store each sample under an enumerated file name


def store_as_torch_tensor(slc, num):
    pt_array = torch.tensor(slc, dtype=torch.float)
    torch.save(pt_array, 'data/id-{}.pt'.format(num))
    # print(slc.size, 'data/id-{}.pt'.format(num), pt_array)
    return num + 1


# Commence file name enumeration with....
ID = 1

# Save to disk
for row in Xy:
    ID = store_as_torch_tensor(row, ID)


# Working with file NAMES of each sample
samples = []
# Walk the 'data' directory
for (dirpath, dirnames, filenames) in os.walk('data'):
    samples = filenames

# Shuffle before splitting
shuffle(samples)

# Split dataset by percentage
train = samples[0: int(len(samples) * .75)]
valid = samples[int(len(samples) * .75):]
print('Training   set: ', len(train))
print('Validation set: ', len(valid))

# Assign sample NAMES to partition dictionary
partition = {'train': train, 'validation': valid}
# print(partition)

# Assign label VALUES to labels dictionary
labels = {}
for sample in samples:
    sample_tensor = torch.load('data/' + sample)
    labels[sample] = int(sample_tensor[-1].item())
# print(labels)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # From file load the data for this sample
        X = torch.load('data/' + ID)
        # Slice off the labels column so that only features are assigned to X
        X = X[:-1]
        # Labels dictionary was loaded into memory earlier; here the
        # file name is used as the key to retrieve the class label value
        y = torch.tensor(self.labels[ID])
        # Apply one-hot encoding to the class label
        #y_ohe = torch.tensor([y])
        y_ohe = y
        #y_ohe = torch.nn.functional.one_hot(y, num_classes=1).float()

        return X, y_ohe


# generate training and test data
# Parameters
params = {'batch_size': 60,  # specify minibatchsize!
          'shuffle': True,
          'num_workers': 0}

params2 = {'batch_size': 20,
           'shuffle': True,
           'num_workers': 0}

params_full_training = {'batch_size': 600,
                        'shuffle': False,
                        'num_workers': 0}

params_full_validation = {'batch_size': 200,
                          'shuffle': False,
                          'num_workers': 0}

# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(
    training_set, **params)  # dataloader for training
training_generator_full = data.DataLoader(training_set, **params_full_training)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(
    validation_set, **params2)  # dataloader for test/validation
validation_generator_full = data.DataLoader(
    validation_set, **params_full_validation)


def gen_spiral_dataset(batchsize,N, r0, circles):
    '''
    generates train and testdataloader for the spiral dataset. additionally it outputs also all input and output data needed to plot the decisionboundary of a model with this data on top.
    Spiral points are 2d point which belong to one of the two possible spiral classes.

    Args:
    batchsize: (int) batchsize of the train (and test) dataloader
    N: (int) number of data per class
    r0: (\in [0,1]) indicates how close the spirals ar at the center. smaller values lead to more complex classification tasks.
    circles (value>0): number of circles which each spiral makes (can also be non-interger) 

    Returns:
    train_dataloader: pytorch dataloader for training (75% of data)
    test_dataloader: pytorch dataloader for testing (25% of data)
    data_X
    data_y
    
    '''
    D = 2
    K = 2
    number = N

    def spiral_xy(i,spiral_num, number, circles = circles, r0 = r0 ):
        '''
        '''
        phi = torch.tensor(0.) # start value of phi
        delta_phi = torch.tensor(2 * np.pi * circles / number) # increment which is added to phi for each new datapoint
        phi= phi+i*delta_phi
        
        r = r0 # initial radius
        delta_r = circles / number
        r = r +i* delta_r

        x= r*spiral_num*torch.cos(phi)
        y = r*spiral_num*torch.sin(phi)
        return [x,y]




    def spiral(spiral_num):
        return [spiral_xy(i, spiral_num,number=N, circles=circles, r0=r0) for i in range(N)]


    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')
    y[0:N] = np.zeros(N)  # represents -1
    y[N:] = np.ones(N)



    X[0:N] = spiral(-1)
    X[N:] = spiral(1)


    data_X = X
    data_y = y

    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("Running on : ", torch.cuda.device_count(), " GPUs!"
            if torch.cuda.device_count() > 1 else " GPU!")
    else:
        print("Running on CPU!")


    # create the data directory
    print('Current working directory: ', os.getcwd())
    if os.path.exists('data'):
        import shutil
        shutil.rmtree('data')
        print('Removed previously created data directory & contents')

    os.mkdir('data')
    print('New data directory created')

    print(os.listdir())

    # Join class labels vector onto features matrix ready for storage
    Xy = np.block([[X, np.reshape(y, (len(y), 1))]])

    # Method to store each sample under an enumerated file name


    def store_as_torch_tensor(slc, num):
        pt_array = torch.tensor(slc, dtype=torch.float)
        torch.save(pt_array, 'data/id-{}.pt'.format(num))
        # print(slc.size, 'data/id-{}.pt'.format(num), pt_array)
        return num + 1


    # Commence file name enumeration with....
    ID = 1

    # Save to disk
    for row in Xy:
        ID = store_as_torch_tensor(row, ID)


    # Working with file NAMES of each sample
    samples = []
    # Walk the 'data' directory
    for (dirpath, dirnames, filenames) in os.walk('data'):
        samples = filenames

    # Shuffle before splitting
    shuffle(samples)

    # Split dataset by percentage
    train = samples[0: int(len(samples) * .75)]
    valid = samples[int(len(samples) * .75):]
    print('Training   set: ', len(train))
    print('Validation set: ', len(valid))

    # Assign sample NAMES to partition dictionary
    partition = {'train': train, 'validation': valid}
    # print(partition)

    # Assign label VALUES to labels dictionary
    labels = {}
    for sample in samples:
        sample_tensor = torch.load('data/' + sample)
        labels[sample] = int(sample_tensor[-1].item())
    # print(labels)


    class Dataset(data.Dataset):
        'Characterizes a dataset for PyTorch'

        def __init__(self, list_IDs, labels):
            'Initialization'
            self.labels = labels
            self.list_IDs = list_IDs

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            # From file load the data for this sample
            X = torch.load('data/' + ID)
            # Slice off the labels column so that only features are assigned to X
            X = X[:-1]
            # Labels dictionary was loaded into memory earlier; here the
            # file name is used as the key to retrieve the class label value
            y = torch.tensor(self.labels[ID])
            # Apply one-hot encoding to the class label
            #y_ohe = torch.tensor([y])
            y_ohe = y
            #y_ohe = torch.nn.functional.one_hot(y, num_classes=1).float()

            return X, y_ohe


    # generate training and test data
    # Parameters
    params = {'batch_size': batchsize,  # specify minibatchsize!
            'shuffle': True,
            'num_workers': 0}

    params2 = {'batch_size': int(batchsize/3),
            'shuffle': True,
            'num_workers': 0}

    # Generators
    training_set = Dataset(partition['train'], labels)
    training_generator = data.DataLoader( training_set, **params)

    validation_set = Dataset(partition['validation'], labels)
    validation_generator = data.DataLoader(
        validation_set, **params2)  # dataloader for test/validation




    return training_generator, validation_generator, data_X, data_y




def plot_decision_boundary(model, features, labels, save_plot=None):
    '''
    plots decisionboundary of the given model. This works only for models with 2-dimensional input!

    Args:
      model: pytorch model
      features: torch tensor of model input data
      labels: torch tensor of corresponding model output data

      Example: features=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
               labels= torch.tensor([[0],[1],[1],[0]])

    Out:
       no return value. generates a plot.
    '''
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = features[:, 0].min()-1, features[:, 0].max()+1
    y_min, y_max = features[:, 1].min()-1, features[:, 1].max()+1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 500  # 250

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # Pass data to predict method
    data_t = torch.tensor(data, dtype=torch.float).to(device)
    # Set model to evaluation mode
    model.eval()
    Z = model(data_t)
    # print(Z.shape)

    # Convert PyTorch tensor to NumPy for plotting.
    # if both values are equal, the the first entry is the argmax, i.e. here 0
    Z_cat = np.argmax(Z.detach().cpu().numpy(), axis=1)
    Z_max_val = np.max(Z.detach().cpu().numpy(), axis=1)
    # Z = Z.detach().cpu().numpy()[:,1] # displays values of output in y axis
    Z_cat = Z_cat.reshape(XX.shape)
    Z_max_val = Z_max_val.reshape(XX.shape)
    # print(Z.shape)
    # print(Z[0,:])
    # print(Z[:,0])
    # fig = plt.figure()
    plt.contourf(XX, YY, Z_cat, cmap='gray', alpha=0.8)  # plt.cm.Spectral
    # sns.countplot(Z_max_val, hue = Z_cat)
    plt.scatter(features[:, 0], features[:, 1],
                c=labels, s=40, cmap=plt.cm.Spectral)
    plt.xlim(XX.min(), XX.max())
    plt.ylim(YY.min(), YY.max())
    if save_plot is not None:
        # save_plot provides the path
        # plt.close()
        plt.savefig(save_plot, bbox_inches='tight')
        plt.close()
        # os.unlink(save_plot)
    else:
        plt.show()
    # fig.savefig('spiral_linear.png')




X_small = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
y_small = torch.tensor([[0], [1], [1], [0]])

def check_accuracy(model, x_data, y_data):
    X_check = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_check = torch.tensor(y_data, dtype=torch.float32).to(device)

    # Set model to evaluation mode
    model.eval()

    # Make predictions
    predictions = model(X_check)
    ''' max_vals is a tensor of probability values.
      arg_maxs is a tensor of the index locations at which
      the maximum probability occured in the tensor.'''

    (max_vals, arg_maxs) = torch.max(predictions.data, dim=1)
    (y_max_vals, y_arg_maxs) = torch.max(y_check.data, dim=1)

    # print('True Labels: {}'.format(y_arg_maxs))
    # print('Predictions: {}'.format(arg_maxs))

    # arg_maxs is tensor of indices [0, 1, 0, 2, 1, 1 . . ]
    num_correct = torch.sum(y_arg_maxs == arg_maxs)
    acc = (num_correct * 100.0 / len(y_data))
    return acc.item()
