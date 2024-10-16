import librosa
import torch
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Time stretch augmentation using librosa.

        Args:
            rate (float): Rate of time-stretching. Values less than 1.0 slow down the audio,
                          values greater than 1.0 speed it up.
        """
        super().__init__()
        self.rate = kwargs.get("rate")

    def __call__(self, data: Tensor):
        """
        Apply time stretch to the input audio tensor.

        Args:
            data (Tensor): Input audio tensor of shape (batch_size, num_samples).

        Returns:
            Tensor: Time-stretched audio tensor of the same shape as input.
        """
        stretched_data = []
        for sample in data:
            sample_np = sample.cpu().numpy()
            stretched_sample = librosa.effects.time_stretch(sample_np, rate=self.rate)
            stretched_sample_tensor = torch.tensor(
                stretched_sample, dtype=sample.dtype, device=sample.device
            )

            if stretched_sample_tensor.shape[0] < sample.shape[0]:
                padded_sample = nn.functional.pad(
                    stretched_sample_tensor,
                    (0, sample.shape[0] - stretched_sample_tensor.shape[0]),
                )
            else:
                padded_sample = stretched_sample_tensor[: sample.shape[0]]

            stretched_data.append(padded_sample)
        return torch.stack(stretched_data)
