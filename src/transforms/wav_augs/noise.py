import torch
from torch import Tensor, nn
from torch.optim.optimizer import Kwargs


class AddGaussianNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Add Gaussian noise to the input audio data.

        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        super().__init__()
        self.mean = kwargs.get("mean")
        self.std = kwargs.get("std")

    def __call__(self, data: Tensor):
        """
        Apply Gaussian noise to the input audio tensor.

        Args:
            data (Tensor): Input audio tensor of shape (batch_size, num_samples).

        Returns:
            Tensor: Audio tensor with added Gaussian noise.
        """
        noise = torch.distributions.Normal(self.mean, self.std).sample(data.shape)
        return data + noise
