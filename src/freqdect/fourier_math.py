import torch
import numpy as np


def batch_fourier_preprocessing(
    image_batch, eps=1e-12, log_scale=False
):
    """Preprosess image batches by computing the Fourier-representation.

    The raw as well as an absolute log scaled version can be computed.

    Args:
        image_batch (np.array): An image of shape (B, H, W, C)
        eps: A small number to stabilize the logarithm.
        log_scale: Use log-scaling if True.
                   Log-scaled coefficients aren't invertible.
                   Default: False.

    Returns:
        [np.array]: The wavelet packets [B, H, W, C].
    """
    image_batch = torch.from_numpy(image_batch.astype(np.float32)).cuda()
    # transform to from H, W, C to C, H, W
    channels = []
    for channel in range(image_batch.shape[-1]):
            channels.append(torch.fft.fft2((image_batch[..., channel])))
    freq = torch.stack(channels, -1)
    del channels
    if log_scale:
        freq = torch.abs(freq)
        freq = torch.log(freq + eps)
    else:
        freq = torch.cat(
            [torch.real(freq),
             torch.imag(freq)], -1)
    return freq.cpu().numpy()