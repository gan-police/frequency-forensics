"""Module implementing wavelet related math functions.

The idea is to provide functionality to make the packet transform useful
for image analysis and gan-content recognition.
"""

from itertools import product
from typing import Optional

import numpy as np
import ptwt
import pywt
import torch


def compute_packet_rep_2d(
    image, wavelet_str: str = "haar", max_lev: int = 3
) -> np.array:
    """Numpy based computation of a 2d full-packet representation.

    Args:
        image (np.aray): Image of shape [height, width].
        wavelet_str (str, optional): The wavelet to use. Defaults to "haar".
        max_lev (int, optional): The number of levels in the representation.
            Defaults to 3.

    Returns:
        np.array: A ready to plot wavelet packet image.
    """
    wavelet = pywt.Wavelet(wavelet_str)
    wp_tree = pywt.WaveletPacket2D(data=image, wavelet=wavelet, mode="reflect")
    # Get the full decomposition
    wp_keys = list(product(["a", "h", "v", "d"], repeat=max_lev))
    count = 0
    img_rows = None
    img = []
    for node in wp_keys:
        packet = np.squeeze(wp_tree["".join(node)].data)
        if img_rows is not None:
            img_rows = np.concatenate([img_rows, packet], axis=1)
        else:
            img_rows = packet
        count += 1
        if count >= np.sqrt(len(wp_keys)):
            count = 0
            img.append(img_rows)
            img_rows = None

    img_pywt = np.concatenate(img, axis=0)
    return img_pywt


def compute_pytorch_packet_representation_2d_image(
    pt_data, wavelet_str: str = "db5", max_lev: int = 5
):
    """Create a packet image to plot."""
    wavelet = pywt.Wavelet(wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket2D(data=pt_data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    wp_keys = list(product(["a", "h", "v", "d"], repeat=max_lev))
    count = 0
    img_pt = []
    img_rows_pt = None
    for node in wp_keys:
        packet = torch.squeeze(ptwt_wp_tree["".join(node)], axis=1)
        if img_rows_pt is not None:
            img_rows_pt = torch.cat([img_rows_pt, packet], axis=2)
        else:
            img_rows_pt = packet
        count += 1
        if count >= np.sqrt(len(wp_keys)):
            count = 0
            img_pt.append(img_rows_pt)
            img_rows_pt = None

    wp_pt = torch.cat(img_pt, axis=1)
    return wp_pt


def compute_pytorch_packet_representation_2d_tensor(
    pt_data: torch.Tensor,
    wavelet_str: Optional[str] = "db5",
    max_lev: int = 5,
    mode: str = "reflect",
) -> torch.Tensor:
    """Compute the wavelet packet representation tensor for a batch of input images.

    Args:
        pt_data: Image tensor of shape [batch, height, width]
        wavelet_str: Wavelet description string. Must be Pywt compatible. Defaults to "db5".
        max_lev: The maximum decomposition level to compute. Defaults to 5.
        mode: The desired boundary treatment approach. Choose zero, reflect or
            boundary. Defaults to reflect.

    Returns:
    : The packet tensor of shape [batch_size, packet_no, packet_height, packet_width]
    """
    wavelet = pywt.Wavelet(wavelet_str)
    # print('wavelet', wavelet_str)
    ptwt_wp_tree = ptwt.WaveletPacket2D(data=pt_data, wavelet=wavelet, mode=mode)

    # get the pytorch decomposition
    # batch_size = pt_data.shape[0]
    wp_keys = list(
        product(
            ["a", "h", "v", "d"],
            repeat=max_lev,
        )
    )
    packet_list = []
    for node in wp_keys:
        packet = torch.squeeze(ptwt_wp_tree["".join(node)], axis=1)
        packet_list.append(packet)

    wp_pt = torch.stack(packet_list, axis=1)
    return wp_pt


def batch_packet_preprocessing(
    image_batch, wavelet="db1", max_lev=3, eps=1e-12, log_scale=False, mode="reflect"
):
    """Preprosess image batches by computing the wavelet packet representation.

    The raw as well as an absolute log scaled version can be computed.

    Args:
        image_batch (np.array): An image of shape (B, H, W, C)
        wavelet (str, optional): A pywt-compatible wavelet string.
            Defaults to 'db1'.
        max_lev (int, optional): The number of decomposition scales
            to use. Defaults to 3.
        eps: A small number to stabilize the logarithm.
        log_scale: Use log-scaling if True.
                   Log-scaled coefficients aren't invertible.
                   Default: False.
        mode: The boundary treatment method. Defaults to reflect.

    Returns:
        [np.array]: The wavelet packets [B, N, H, W, C].
    """
    image_batch = torch.from_numpy(image_batch.astype(np.float32)).cuda()
    # transform to from H, W, C to C, H, W
    channels = []
    for channel in range(image_batch.shape[-1]):
        with torch.no_grad():
            channel_packets = compute_pytorch_packet_representation_2d_tensor(
                image_batch[:, :, :, channel],
                wavelet_str=wavelet,
                max_lev=max_lev,
                mode=mode,
            )
        channels.append(channel_packets)
    packets = torch.stack(channels, -1)
    del channels
    if log_scale:
        packets = torch.abs(packets)
        packets = torch.log(packets + eps)
    return packets.cpu().numpy()


def identity_processing(image_batch):
    """Return the input unchanged."""
    return image_batch
