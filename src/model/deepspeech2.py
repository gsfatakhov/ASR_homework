from torch import nn
from torch.nn import Sequential


class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 Model
    """

    def __init__(self, n_feats, n_tokens, rnn_hidden=512, num_rnn_layers=5):
        """
        Args:
            n_feats (int): number of input features (e.g., number of frequency bins).
            n_tokens (int): number of tokens in the vocabulary.
            rnn_hidden (int): number of hidden units in each RNN layer.
            num_rnn_layers (int): number of RNN layers.
        """
        super().__init__()

        self.conv = Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1
            ),  # output: (batch, 32, time/2, freq/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # output: (batch, 64, time/4, freq/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=64 * (n_feats // 4),
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram of shape (batch, time, features).
            spectrogram_length (Tensor): original lengths of the spectrograms.
        Returns:
            output (dict): output dict containing log_probs and transformed lengths.
        """
        x = spectrogram.unsqueeze(1)

        x = self.conv(x)

        # (batch, channels, time, features) -> (batch, time, channels * features)
        batch_size, channels, time, features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, time, -1)

        x, _ = self.rnn(x)

        x = self.fc(x)
        log_probs = nn.functional.log_softmax(x, dim=-1)

        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    @staticmethod
    def transform_input_lengths(input_lengths):
        """
        Adjust the input lengths based on the convolutional downsampling.
        Args:
            input_lengths (Tensor): original input lengths
        Returns:
            output_lengths (Tensor): adjusted lengths after downsampling
        """
        for _ in range(2):
            input_lengths = (input_lengths + 1) // 2
        return input_lengths // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info
