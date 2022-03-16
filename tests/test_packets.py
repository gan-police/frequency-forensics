"""Ensure ptwt and pywt packets are equivalent."""

import sys
from itertools import product

import numpy as np
import pytest
import pywt
import torch
from scipy import misc

sys.path.append("./src")
from src.freqdect.wavelet_math import compute_pytorch_packet_representation_2d_tensor


def compute_pywt_packet_representation_2d_tensor(
    data, wavelet_str: str = "db5", max_lev: int = 5
):
    """To Ensure pywt and ptwt equivalence compute pywt packets."""
    wavelet = pywt.Wavelet(wavelet_str)
    pywt_wp_tree = pywt.WaveletPacket2D(data=data, wavelet=wavelet, mode="reflect")

    # get the pytorch decomposition
    # batch_size = pt_data.shape[0]
    wp_keys = list(product(["a", "h", "v", "d"], repeat=max_lev))
    packet_list = []
    for node in wp_keys:
        packet = pywt_wp_tree["".join(node)].data
        packet_list.append(packet)

    wp_py = np.stack(packet_list, axis=0)
    return wp_py


@pytest.mark.slow
def test_packets():
    """Runs the pywt ptwt comparison test."""
    face = misc.face()[256:512, 256:512]
    grey_face = np.mean(face, axis=-1).astype(np.float64)
    # add batch dimension.
    pt_face = torch.unsqueeze(torch.from_numpy(grey_face), 0)
    py_packets = compute_pywt_packet_representation_2d_tensor(
        pt_face.squeeze(0).numpy(), "haar"
    )
    pt_packets = (
        compute_pytorch_packet_representation_2d_tensor(pt_face, "haar")
        .squeeze(0)
        .numpy()
    )
    assert np.allclose(py_packets, pt_packets)  # noqa: S101


if __name__ == "__main__":
    """Run the test."""
    test_packets()
