from model import EEGModel, CNN1D, RegionalTransformer
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

config = {}

# m = EEGModel(config)


def generate_synthetic_eeg_data(
    batch_size, n_channels, sequence_length, y_value, noise_amount=0.1
):

    return (
        torch.rand(batch_size, n_channels, sequence_length) - 0.5
    ) * noise_amount + y_value**2 * np.linspace(0, 1, sequence_length)


class TestingModel1(nn.Module):
    def __init__(self, config, output_size):
        super(TestingModel1, self).__init__()
        self.config = config
        self.cnn1d = CNN1D(
            config["sequence_length"],
            config["convolution_dimension_length"],
            config["kernel_size"],
            config["n_1d_cnn_layers"],
            config["n_channels"],
            config["dropout"],
        )
        self.regional_transformer = RegionalTransformer(
            config["input_dim"],
            config["num_heads"],
            config["ff_dim"],
            config["num_layers"],
            config["sequence_length"] - 6,
            config["latent_dim"],
            config["dropout"],
            config["verbose"],
        )

        self.feed_forward = torch.nn.Linear(
            config["convolution_dimension_length"]
            * config["n_channels"]
            * config["latent_dim"],
            output_size,
        )

    def forward(self, x):
        bath_size, _, _ = x.shape
        x = self.cnn1d(x)
        x = self.regional_transformer(x)
        x = x.view(bath_size, -1)
        x = self.feed_forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)


def test_cnn1d() -> CNN1D:
    batch_size = 10  # Number of samples in a batch
    sequence_length = 1000  # Number of sampled points per channel
    n_channels = 8  # Number of EEG channels
    kernel_size = 3  # Size of the convolution kernel
    n_1d_cnn_layers = 3  # Number of convolution layers
    convolution_dimension_length = 64  # Number of filters in convolution layers

    # Create a synthetic EEG data batch
    synthetic_eeg_data = torch.randn(batch_size, n_channels, sequence_length)
    plt.plot(synthetic_eeg_data[0, 0, :].detach().numpy())

    # Initialize the CNN1D model
    print("sequence length", sequence_length)
    print("convolution_dimension_length", convolution_dimension_length)
    print("kernel_size", kernel_size)
    print("n_1d_cnn_layers", n_1d_cnn_layers)
    print("n_channels", n_channels)
    cnn1d_model = CNN1D(
        sequence_length,
        convolution_dimension_length,
        kernel_size,
        n_1d_cnn_layers,
        n_channels,
    )

    # Pass the synthetic data through the model
    output = cnn1d_model(synthetic_eeg_data)
    print("output shape", output.shape)
    # Check the output shape
    assert output.shape == (
        batch_size,
        n_channels,
        sequence_length - 2 * n_1d_cnn_layers,
        convolution_dimension_length,
    ), "Output shape mismatch"
    return cnn1d_model


def test_regional_tranformer():
    one_d_cnn = test_cnn1d()
    # Initialize the RegionalTransformer model
    input_dim = 64  # Dimensionality of input features for the transformers
    num_heads = 4  # A standard choice for the number of heads in multi-head attention
    ff_dim = 256  # Feedforward dimension in transformers
    num_layers = 2  # Number of transformer layers
    sequence_length = 1000  # Number of sampled points per channel
    latent_dim = 128  # Dimensionality of the latent space

    regional_transformer = RegionalTransformer(
        input_dim,
        num_heads,
        ff_dim,
        num_layers,
        sequence_length - 6,
        latent_dim,
        dropout=0.1,
        verbose=True,
    )
    synthetic_eeg_data = generate_synthetic_eeg_data(10, 8, 1000, 0.25, 0.1)
    output = one_d_cnn(synthetic_eeg_data)
    print("output shape", output.shape)
    output = regional_transformer(output)
    print("output shape", output.shape)


def get_basic_ass_regional_transformer():

    config = {
        "sequence_length": 100,
        "convolution_dimension_length": 8,
        "kernel_size": 3,
        "n_1d_cnn_layers": 3,
        "n_channels": 8,
        "dropout": 0.1,
        "input_dim": 64,
        "num_heads": 4,
        "ff_dim": 16,
        "num_layers": 2,
        "latent_dim": 4,
        "verbose": 0,
        "output_size": 1,
    }
    model = TestingModel1(config, config["output_size"])
    return model

def test_regional_tranformer_frfr():
    model = get_basic_ass_regional_transformer()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100  # Number of training epochs
    batch_size = 32  # Batch size for training

    try:
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            total_loss = 0.0

            # Simulating training batches - replace this with actual data loading if necessary
            for _ in range(100):  # Assuming 100 batches per epoch
                y_value = np.random.rand(
                    batch_size
                )  # A constant value for the synthetic data
                inputs = generate_synthetic_eeg_data(
                    batch_size,
                    config["n_channels"],
                    config["sequence_length"],
                    y_value,
                    0.1,
                )

                targets = y_value * torch.ones(batch_size, config["output_size"])

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Logging
            average_loss = total_loss / 100
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    except KeyboardInterrupt:
        print("Interrupted")

    # Evaluate the model

    y_value = random.random()
    generated_data = generate_synthetic_eeg_data(10, 8, 100, y_value, 0.1)
    predicted = model(generated_data)
    simulated = y_value * torch.ones(1, config["output_size"])
    # plt.plot(generated_data[0, 0, :].detach().numpy(), label="test_data")
    # plt.plot(
    #     # np.linspace(0, 1, 100),
    #     np.linspace(0, 1, 100) * simulated.squeeze(0).detach().numpy(),
    #     label="y_pred",
    # )
    # plt.legend()

    print("predicted", predicted)
    print("true", y_value)
    plt.show()

def test_model_on_mnist():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

# Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set up data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
    config = {
        "sequence_length": 28,
        "convolution_dimension_length": 8,
        "kernel_size": 3,
        "n_1d_cnn_layers": 3,
        "n_channels": 28,
        "dropout": 0.1,
        "input_dim": 64,
        "num_heads": 4,
        "ff_dim": 16,
        "num_layers": 2,
        "latent_dim": 4,
        "verbose": 0,
        "output_size": 10,
    }
    model = TestingModel1(config, config["output_size"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
    epochs = 1
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.squeeze(1)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}')

# Save the trained model if needed
# torch.save(model.state_dict(), 'mnist_model.pth')


test_model_on_mnist()
