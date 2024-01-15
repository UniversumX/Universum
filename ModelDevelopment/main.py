from model import EEGModel, CNN1D, RegionalTransformer
import torch

config = {}

# m = EEGModel(config)


def test_cnn1d() -> CNN1D:
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
    )
    synthetic_eeg_data = torch.randn(10, 8, 1000)
    output = one_d_cnn(synthetic_eeg_data)
    print("output shape", output.shape)
    output = regional_transformer(output)
    print("output shape", output.shape)


test_regional_tranformer()
