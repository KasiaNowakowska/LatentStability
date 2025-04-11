import matplotlib as mpl
from pathlib import Path
import torch
import torchinfo
import time
import einops
import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cpu")

sys.path.append('../')
from neuralnetwork.autoencoder import CAE
from neuralnetwork.preprocessing import normalize_data, train_valid_test_split
from neuralnetwork.earlystopping import EarlyStopper
from neuralnetwork.losses import LossTracker

def load_data_set(file, names, snapshots):
    with h5py.File(file, 'r') as hf:
        print(hf.keys())

        time_vals = hf['total_time_all'][:snapshots]  # 1D time vector

        data = np.zeros((len(time_vals), len(x), len(z), len(names)))
        
        index=0
        for name in names:
            print(name)
            print(hf[name])
            Var = np.array(hf[name])
            data[:,:,:,index] = Var[:snapshots,:,0,:]
            index+=1

    return data, time_vals

#### LOAD DATA ####
data_input_path = '../../input_data/'
variables = ['q_all']
names = ['q']
num_variables = 1
x = np.load(data_input_path+'/x.npy')
z = np.load(data_input_path+'/z.npy')
snapshots =20000
data_set, time_vals = load_data_set(data_input_path+'/data_4var_5000_30000.h5', variables, snapshots)
print('shape of dataset', np.shape(data_set))

# Define some parameters for the data
ks_data = {
    'L': 22,
    'N_data': 20000,
    'N_trans': 100,
    'dt': 2,
    'Nx': 256,
    'Nz': 64,
    'train_ratio': 0.5,
    'valid_ratio': 0.2,
    'batchsize': 128,
    'normtype': "maxmin", 
    'lyap': 30, 
    'upsample':1,
    'add_noise': False
}

print(np.shape(data_set))

def add_gaussian_noise(inputs, mean=0.0, std=0.05):
    noise = torch.randn_like(inputs) * std + mean
    return inputs + noise

# Parameters
sampling_rate = 256  # Number of samples (dimensions)
frequency = 1  # Frequency of the sine wave in Hz
duration = 1  # Duration in seconds

# Time vector
t = np.linspace(0, duration, sampling_rate)

# Generate the sine wave
sine_wave = np.sin(2 * np.pi * frequency * t)

U = data_set[:, :, 32, :]
U = einops.rearrange(U, 't h c-> t c h')
sysdim = U.shape[0]
print(f"Shape of KS data: {U.shape}")
add_noise = ks_data['add_noise']

# Normalize and split
U_normalized, U_max, U_min = normalize_data(U, normtype=ks_data['normtype'])
U_train_series, U_valid_series, U_test_series = train_valid_test_split(U_normalized, ks_data)
sine_wave = np.expand_dims(sine_wave, 0)  # Shape becomes (1, 10)
sine_wave = np.expand_dims(sine_wave, 0)  # Shape becomes (1, 1, 10)
#U_train_series = sine_wave
U_train_series = U_train_series[::500,...]
U_valid_series = U_train_series
U_test_series = U_train_series
print(U_train_series.shape, U_valid_series.shape, U_test_series.shape)

train_loader = torch.utils.data.DataLoader(torch.from_numpy(U_train_series).float(), batch_size=ks_data['batchsize'])
valid_loader = torch.utils.data.DataLoader(torch.from_numpy(U_valid_series).float(), batch_size=ks_data['batchsize'])
#print('train_loader shape', train_loader.shape)

cae_model = CAE(16)
cae_model = cae_model.to(device)
print(torchinfo.summary(cae_model, input_size=(1, 1, ks_data['Nx'])))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cae_model.parameters(), lr=0.001)

# Define the number of epochs and the gamma parameter for the scheduler
epochs = 1000
gamma = 0.99

# Create an instance of ExponentialLR and associate it with your optimizer
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
loss_tracker = LossTracker(len(train_loader), len(valid_loader))

early_stopper = EarlyStopper(patience=200, min_delta=1e-6)
weighing_dissipation = 1
cae_model = cae_model.to(device)
# wandb.watch(model)
best_model_state_dict = cae_model.state_dict()  # Save the best model state_dict
patience = 100
training_losses_per_epoch = []
validation_losses_per_epoch = []
best_validation_loss = float('inf')

for epoch in range(epochs):
    loss_tracker.set_start_time(time.time())
    loss_tracker.reset_current_loss()
    print(optimizer.param_groups[0]['lr'])

    # Training loop
    cae_model.train()
    for step, x_batch_train in enumerate(train_loader):
        x_batch_train = x_batch_train.to(device)

        optimizer.zero_grad()
        encoded, output = cae_model(x_batch_train.to(device))
        print('shape of output', output.shape)
        print('shape of train data', x_batch_train.shape)
        loss = criterion(output, x_batch_train)
        loss.backward()
        optimizer.step()
        loss_tracker.update_current_loss('training', loss)
    loss_tracker.print_current_loss(epoch, 'training')

    loss_tracker.set_start_time(time.time())
    scheduler.step()
    # Validation loop
    cae_model.eval()
    with torch.no_grad():
        for valid_step, x_batch_valid in enumerate(valid_loader):
            x_batch_valid = x_batch_valid.to(device)
            encoded, output = cae_model(x_batch_valid)
            loss = criterion(output, x_batch_valid)
            loss_tracker.update_current_loss('validation', loss)

    loss_tracker.print_current_loss(epoch, 'validation')
    loss_tracker.calculate_and_store_average_losses()

    if loss_tracker.check_best_validation_loss():
        early_stopper.reset_counter()
        best_model_state_dict = cae_model.state_dict()  # the best model state_dict
        torch.save(best_model_state_dict, 'best_model.pth')  # Save the best model
        print('Saved best model')
    if early_stopper.track(loss_tracker.get_current_validation_loss()):
        break

loss_tracker.create_loss_plot2(modelpath=Path("./"))


snapshot_train = torch.from_numpy(U_train_series).float()
encoded, snapshot_decoded = cae_model(snapshot_train.to(device))

data1 = snapshot_train.numpy(force=True)[0, 0, :]
data2 = snapshot_decoded.numpy(force=True)[0, 0, :]

fig, ax = plt.subplots(1)
ax.plot(data1, label='true')
ax.plot(data2, label='pred')
plt.legend()
fig.savefig("./plot1.png", dpi=100)

data1 = snapshot_train.numpy(force=True)[1, 0, :]
data2 = snapshot_decoded.numpy(force=True)[1, 0, :]

fig, ax = plt.subplots(1)
ax.plot(data1, label='true')
ax.plot(data2, label='pred')
plt.legend()
fig.savefig("./plot2.png", dpi=100)


# ### training data ###
# N_lyap = ks_data["lyap"]
# N_plot = 10*N_lyap//ks_data["dt"]
# fs = 14
# cmap = 'RdBu_r'

# fig, axs = plt.subplots(3,1 , figsize=(12, 9), sharey=True)
# snapshot = torch.from_numpy(U_train_series[:N_plot]).float()
# encoded, snapshot_decoded = cae_model(snapshot.to(device))

# # Extract data
# data1 = snapshot.numpy(force=True)[:N_plot, 0, :]
# data2 = snapshot_decoded.numpy(force=True)[:N_plot, 0, :]
# data3 = data1 - data2

# lyapunov_time = ks_data["lyap"] * np.arange(0, 1000, (ks_data["dt"]*ks_data["upsample"]))

# for i, data in enumerate([data1, data2, data3]):
#     ax = axs[i]
#     c1=ax.pcolormesh(lyapunov_time[:N_plot], x, data.T, cmap=cmap)
#     ax.set_ylabel('x')
#     cbar = fig.colorbar(c1, ax=ax)

# axs[0].set_xlabel(r'$\tau_{\lambda}$', fontsize=fs)
# axs[0].set_title("Reference")
# axs[1].set_title("CAE")
# axs[2].set_title("Error")
# # Show or save plot
# plt.tight_layout()
# fig.savefig("./cae_train.png", dpi=100)



# ### unseen data ###
# N_lyap = ks_data["lyap"]
# N_plot = 5*N_lyap//ks_data["dt"]
# fs = 14
# cmap = 'RdBu_r'

# fig, axs = plt.subplots(3,1 , figsize=(12, 9), sharey=True)
# snapshot = torch.from_numpy(U_test_series[:N_plot]).float()
# encoded, snapshot_decoded = cae_model(snapshot.to(device))

# # Extract data
# data1 = snapshot.numpy(force=True)[:N_plot, 0, :]
# data2 = snapshot_decoded.numpy(force=True)[:N_plot, 0, :]
# data3 = data1 - data2

# lyapunov_time = ks_data["lyap"] * np.arange(0, 1000, (ks_data["dt"]*ks_data["upsample"]))

# for i, data in enumerate([data1, data2, data3]):
#     ax = axs[i]
#     c1=ax.pcolormesh(lyapunov_time[:N_plot], x, data.T, cmap=cmap)
#     ax.set_ylabel('x')
#     cbar = fig.colorbar(c1, ax=ax)

# axs[0].set_xlabel(r'$\tau_{\lambda}$', fontsize=fs)
# axs[0].set_title("Reference")
# axs[1].set_title("CAE")
# axs[2].set_title("Error")
# # Show or save plot
# plt.tight_layout()
# fig.savefig("./cae.png", dpi=100)
