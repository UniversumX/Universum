import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """
    Modified EEGFormer 1D CNN implementation using depth-wise convolution.
    TODO this may need to be changed for the actual data
    """

    def __init__(
        self,
        sequence_length: int,
        convolution_dimension_length: int,
        kernel_size: int,
        n_1d_cnn_layers: int,
        n_channels=8,
    ):
        super(CNN1D, self).__init__()
        self.n_channels = n_channels  # Number of channels
        self.sequence_length = sequence_length  # Number of sampled points

        # Ensure at least one layer
        assert n_1d_cnn_layers >= 1, "Number of 1D CNN layers must be at least 1"

        # Initial depth-wise convolution layer
        self.initial_conv = nn.Conv1d(n_channels, n_channels * convolution_dimension_length, kernel_size=kernel_size, groups=n_channels, padding=(kernel_size - 1) // 2)

        # Subsequent depth-wise convolution layers
        self.subsequent_convs = nn.ModuleList([
            nn.Conv1d(n_channels * convolution_dimension_length, n_channels * convolution_dimension_length, kernel_size=kernel_size, groups=n_channels, padding=(kernel_size - 1) // 2)
            for _ in range(1, n_1d_cnn_layers)
        ])

    def forward(self, x):
        """
        Expected input shape: (batch_size, n_channels, sequence_length)
        """
        # Apply initial depth-wise convolution
        output = self.initial_conv(x)

        # Apply subsequent depth-wise convolutions
        for conv in self.subsequent_convs:
            output = conv(output)

        # Reshape the output to maintain the sequence length
        batch_size, _, _ = x.shape
        output = output.view(batch_size, self.n_channels, -1)

        # Ensure output sequence length matches input sequence length
        output = output[:, :, :self.sequence_length]

        return output

# Example usage
# model = CNN1D(sequence_length=1000, convolution_dimension_length=64, kernel_size=3, n_1d_cnn_layers=3, n_channels=8)
# input_data = torch.randn(10, 8, 1000)  # synthetic data
# output = model(input_data)
# print(output.shape)  # should be torch.Size([10, 8, 1000])




# TODO make sure this can generalize
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x

# TODO implement
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# TODO implement
class SynchronousTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(SynchronousTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x):
        # x should be organized to emphasize synchronous aspects
        for layer in self.layers:
            x = layer(x)
        return x



class RegionalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(RegionalTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(input_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x):
        # x should be organized to emphasize regional aspects
        for layer in self.layers:
            x = layer(x)
        return x




class EEGformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(EEGformerEncoder, self).__init__()
        self.temporal_transformer = TemporalTransformer(input_dim, num_heads, ff_dim, num_layers, dropout)
        self.synchronous_transformer = SynchronousTransformer(input_dim, num_heads, ff_dim, num_layers, dropout)
        self.regional_transformer = RegionalTransformer(input_dim, num_heads, ff_dim, num_layers, dropout)

    def forward(self, x):
        x = self.temporal_transformer(x)
        x = self.synchronous_transformer(x)
        x = self.regional_transformer(x)
        return x


class EEGformerDecoderForRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EEGformerDecoderForRegression, self).__init__()
        # Assuming input_dim is the output feature size from the encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # output_dim should match the accelerometer data dimension

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, as this is a regression task
        return x

class EEGFormerForRegression(nn.Module):
    def __init__(self, sequence_length, convolution_dimension_length, kernel_size, n_1d_cnn_layers, n_channels, input_dim, num_heads, ff_dim, num_layers, dropout, hidden_dim, output_dim):
        super(EEGFormerForRegression, self).__init__()
        self.cnn1d = CNN1D(sequence_length, convolution_dimension_length, kernel_size, n_1d_cnn_layers, n_channels)
        self.encoder = EEGformerEncoder(input_dim, num_heads, ff_dim, num_layers, dropout)
        self.decoder = EEGformerDecoderForRegression(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        # Apply CNN1D for feature extraction
        x = self.cnn1d(x)
        # Reshape x to fit the encoder input if necessary
        x = x.permute(2, 0, 1)  # Assuming we need to permute to (sequence_length, batch, features)
        # Apply the encoder
        x = self.encoder(x)
        # Apply the decoder
        x = self.decoder(x)
        return x



# alternative models, LSTM based, we should add GRU as well
class EEG2AccelModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, output_dim):
        super(EEG2AccelModel, self).__init__()
        # CNN for EEG feature extraction
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # LSTM for time series prediction
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply CNN layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)

        # Reshape for LSTM
        x = x.view(x.size(0), -1, x.size(1))  # Reshape input for LSTM: (batch_size, seq_len, features)

        # Apply LSTM layers
        lstm_out, (hidden, _) = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
        return x

class EEGModel(torch.nn.Module):
    def __init__(
        self,
    ):
        super(EEGModel, self).__init__()

    def forward(self, x):
        pass



"""

class CNN1D(nn.Module):

    # This was mostly stolen from the EEGFormer implementation I found on github


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

        #Expected input shape:
        #(batch_size, n_channels, sequence_lengths)
 
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
"""



"""
the EEGformer model specialize in different aspects of the neural signal (temporal, regional, and synchronous) is significantly determined during the dataset preparation and how the data is fed into these transformers. The architecture of the transformers themselves doesn't inherently distinguish between these characteristics; it's the structure and preprocessing of the input data that dictates what each transformer focuses on.

Here's how this specialization typically works in the context of EEG data:

Temporal Transformer
Focus: Processes the temporal aspects of EEG data.
Data Preparation: The input data should emphasize the temporal dynamics. This means organizing the EEG data so that the sequence input to the transformer represents different time points. The transformer will then learn patterns across these temporal sequences.
Synchronous Transformer
Focus: Deals with synchronous patterns of brain activity across different channels.
Data Preparation: The input data should be structured to highlight the synchronous activity across different EEG channels at the same time point. This might involve reorganizing the data so that for each time point, the input features represent the simultaneous readings from different EEG channels.
Regional Transformer
Focus: Handles different brain regions.
Data Preparation: The input data should be prepared in a way that emphasizes spatial (regional) relationships. This could involve structuring the data such that the input to the transformer at each time step represents data from different brain regions.
Implementation Consideration
Dataset Structure: Careful structuring of the input data is crucial. This involves reshaping and organizing the EEG data appropriately before it's fed into each transformer.
Feature Extraction: Initial layers (like the 1DCNN you might use) are responsible for extracting relevant features from raw EEG data, which the transformers then process. The way these features are extracted and organized can greatly influence the focus of each transformer.
Sequential Processing: The EEGformer model processes the data sequentially through these transformers. The output of one becomes the input to the next, adding layers of contextual understanding (temporal, synchronous, regional) at each stage.
Example
Consider an EEG dataset with readings from multiple channels over time. For the Synchronous Transformer, you'd organize the data so that for each time step, the feature vector represents concurrent readings from all channels. For the Regional Transformer, you'd organize the data to focus on spatial patterns, perhaps grouping channels according to their location on the scalp.

In essence, while the transformer architecture is capable of capturing complex relationships in the data, it's the way the data is presented to each transformer that directs its focus towards temporal, synchronous, or regional aspects of the EEG signal.

"""