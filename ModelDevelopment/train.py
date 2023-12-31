import torch
import torch.nn as nn 

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
num_epochs = 5  # adjust as needed

# Create DataLoader
train_loader = get_data_loader(edf_file_path, segment_length, batch_size)

# Initialize your model, loss function, and optimizer
model = EEGFormerForRegression(...)  # Initialize with appropriate parameters
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for eeg_data, accel_data in train_loader:
        # Forward pass
        outputs = model(eeg_data)
        loss = criterion(outputs, accel_data)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

