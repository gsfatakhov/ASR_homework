from torch import Tensor, nn


class Normalize1D(nn.Module):
    def __init__(self, mean: float = 0.5, std: float = 0.5, eps: float = 1e-5):
        """
        Normalize the input along the time and frequency dimensions (assuming [batch, time, frequency]).
        Args:
            mean (float): Desired mean value for normalization.
            std (float): Desired standard deviation for normalization.
            eps (float): A small value to avoid division by zero during normalization.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.
        Args:
            data (Tensor): Input tensor of shape [batch_size, time, frequency].
        Returns:
            Tensor: Normalized tensor.
        """
        data_mean = data.mean(dim=[1, 2], keepdim=True)
        data_std = data.std(dim=[1, 2], keepdim=True) + self.eps
        normalized_data = (data - data_mean) / data_std
        scaled_data = normalized_data * self.std + self.mean
        return scaled_data
