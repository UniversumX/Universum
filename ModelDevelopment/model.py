import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    This was mostly stolen from the EEGFormer implementation I found on github
    """

    def __init__(
        self,
        sequence_length: int,
        convolution_dimension_length: int,
        kernel_size: int,
        n_1d_cnn_layers: int,
        n_channels=8,
    ):
        super().__init__()
        self.n_channels = n_channels  # no. of channels
        self.sequence_length = sequence_length  # no. of sampled points
        self.n_1d_cnn_layers = n_1d_cnn_layers
        assert n_1d_cnn_layers >= 1
        self.conv_layers = [
            nn.Conv1d(1, convolution_dimension_length, kernel_size=kernel_size)
        ]
        for i in range(1, n_1d_cnn_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    convolution_dimension_length,
                    convolution_dimension_length,
                    kernel_size=kernel_size,
                )
            )

    def forward(self, x):
        """
        Expected input shape:
        (batch_size, n_channels, sequence_lengths)
        """
        outputs = []
        # The idea of this loop is that we use the same 1d conv layer on every single channel to extract artifacts from it,
        for i in range(self.n_channels):
            # For every channel in the eeg device
            output_tensor = x[:, i : i + 1, :]
            for layer in self.conv_layers:
                output_tensor = layer(output_tensor)
            outputs.append(output_tensor.unsqueeze(1))

        output_tensor = torch.cat(outputs, dim=1)

        #! If the output_tensor is not at the seuqnece_length length, then pad it with 0's?'
        output_tensor = output_tensor[:, :, : self.sequence_length]
        return output_tensor


class EEGModel(torch.nn.Module):
    def __init__(
        self,
    ):
        super(EEGModel, self).__init__()

    def forward(self, x):
        pass
