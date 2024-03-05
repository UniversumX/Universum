from model import EEGModel, CNN1D, RegionalTransformer
import torch

config = {}

# m = EEGModel(config)
generate_synthetic_eeg_data = torch.rand


class TestingModel1:
    def __init__(self, config, output_size):
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
        print(x.shape)
        x = self.feed_forward(x)
        return x


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
        verbose=True,
    )
    synthetic_eeg_data = generate_synthetic_eeg_data(10, 8, 1000)
    output = one_d_cnn(synthetic_eeg_data)
    print("output shape", output.shape)
    output = regional_transformer(output)
    print("output shape", output.shape)


def test_regional_tranformer_frfr():
    config = {
        "sequence_length": 1000,
        "convolution_dimension_length": 64,
        "kernel_size": 3,
        "n_1d_cnn_layers": 3,
        "n_channels": 8,
        "dropout": 0.1,
        "input_dim": 64,
        "num_heads": 4,
        "ff_dim": 256,
        "num_layers": 2,
        "latent_dim": 128,
        "verbose": True,
    }
    model = TestingModel1(config, 3)
    model.forward(generate_synthetic_eeg_data(10, 8, 1000))


test_regional_tranformer_frfr()
