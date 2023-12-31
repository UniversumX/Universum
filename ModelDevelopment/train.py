import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from model import EEGFormerForRegression
from dataset import EEGAccelDataset
from torch.utils.data import Dataset, DataLoader

def get_data_loader(edf_file_path, segment_length, batch_size, transform=None):
    dataset = EEGAccelDataset(edf_file_path, segment_length, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader



# Assuming 'eeg_data' is your input EEG data and 'accel_data' is your target accelerometer data
# train_loader is a DataLoader object containing your training data
# Parameters
edf_file_path = 'dummy_data/dummy_set.edf'
segment_length = 1000  # example segment length, adjust as needed
batch_size = 8  # adjust as needed
num_epochs = 1  # adjust as needed

# Create DataLoader
train_loader = get_data_loader(edf_file_path, segment_length, batch_size)

# Example parameters for initializing the model
sequence_length = 256  # 1 second of data at 256 Hz sampling rate
convolution_dimension_length = 64  # Assuming a depth of 64 for convolutional features
kernel_size = 3  # A standard choice for kernel size
n_1d_cnn_layers = 3  # Starting with 3 convolutional layers
n_channels = 8  # Assuming the first 8 channels are EEG

input_dim = 64  # Dimensionality of input features for the transformers
num_heads = 4  # A standard choice for the number of heads in multi-head attention
ff_dim = 256  # Feedforward dimension in transformers
num_layers = 2  # Number of transformer layers
dropout = 0.1  # Dropout rate

hidden_dim = 128  # Hidden dimensionality in the decoder
output_dim = 3  # Assuming we are predicting 3D accelerometer data

# Initialize the EEGFormerForRegression model
model = EEGFormerForRegression(
    sequence_length,
    convolution_dimension_length,
    kernel_size,
    n_1d_cnn_layers,
    n_channels,
    input_dim,
    num_heads,
    ff_dim,
    num_layers,
    dropout,
    hidden_dim,
    output_dim
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for eeg_data, accel_data in train_loader:
        # Forward pass
        outputs = model(eeg_data)
        # Inside training loop
        print("Batch EEG data shape:", eeg_data.shape)
        print("Batch Accel data shape before permute:", accel_data.shape)
        #accel_data = accel_data.permute(2, 0, 1)  # Adjusting the target shape to match the output
        print("Batch Accel data shape after permute:", accel_data.shape)
        # accel_data = accel_data.permute(1, 2, 0)  # Adjusting to (batch_size, channels, sequence_length)
        loss = criterion(outputs, accel_data)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for inference
        for eeg_data, actual_accel_data in train_loader:
            # Forward pass to get predictions
            predicted_accel_data = model(eeg_data).cpu()

            # Selecting the first sample in the batch for visualization
            predicted_sample = predicted_accel_data[0].numpy()  # Convert to NumPy array for plotting
            actual_sample = actual_accel_data[0].numpy()

            print("Actual Sample Shape:", actual_sample.shape)
            print("Predicted Sample Shape:", predicted_sample.shape)

            # Assuming the shape is (3, 1000)
            time_points = range(predicted_sample.shape[1])  # time points

            # Create a plot for each dimension of the accelerometer data
            for i in range(3):
                plt.figure(figsize=(10, 4))
                plt.plot(time_points, actual_sample[i], label='Actual', marker='o', linewidth=1, alpha=0.7)
                plt.plot(time_points, predicted_sample[i], label='Predicted', marker='x', linewidth=1, alpha=0.7)
                plt.title(f'Accelerometer Data - Dimension {i+1}')
                plt.xlabel('Time')
                plt.ylabel('Acceleration')
                plt.legend()
                plt.show()


            # Break after the first batch for demonstration
            break





