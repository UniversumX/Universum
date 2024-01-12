from model import EEGModel, CNN1D
import torch

config = {}

# m = EEGModel(config)


def test_cnn1d():
    batch_size = 10  # Number of samples in a batch
    sequence_length = 1000  # Number of sampled points per channel
    n_channels = 8  # Number of EEG channels
    kernel_size = 3  # Size of the convolution kernel
    n_1d_cnn_layers = 3  # Number of convolution layers
    convolution_dimension_length = 64  # Number of filters in convolution layers

    # Create a synthetic EEG data batch
    synthetic_eeg_data = torch.randn(batch_size, n_channels, sequence_length)

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
    ), "Output shape mismatch"


# Run the test
test_cnn1d()
