
import cv2
import torch
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pywt._doc_utils import _2d_wp_basis_coords
from src.wavelet_math import compute_packet_rep_2d, compute_pytorch_packet_representation_2d


def draw_2d_wp_basis(shape, keys, fmt='k', plot_kwargs={}, ax=None,
                     label_levels=0):
    """Plot a 2D representation of a WaveletPacket2D basis.
       Based on: pywt._doc_utils.draw_2d_wp_basis """
    coords, centers = _2d_wp_basis_coords(shape, keys)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    for coord in coords:
        ax.plot(coord[0], coord[1], fmt)
    ax.set_axis_off()
    ax.axis('square')
    if label_levels > 0:
        for key, c in centers.items():
            if len(key) <= label_levels:
                ax.text(c[0], c[1], ''.join(key),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=6)
    return fig, ax


def read_pair(path_real, path_fake):
    face = cv2.cvtColor(cv2.imread(path_real),
                        cv2.COLOR_BGR2RGB)/255.
    fake_face = cv2.cvtColor(cv2.imread(path_fake),
                             cv2.COLOR_BGR2RGB)/255.
    return face, fake_face


def compute_packet_rep_img(image, wavelet_str, max_lev):

    if len(image.shape) == 3:
        channels_lst = []
        for channel in range(3):
            channels_lst.append(
                compute_packet_rep_2d(image[:, :, channel],
                                      wavelet_str, max_lev))
        return np.stack(channels_lst, axis=-1)
    else:
        assert len(image.shape) == 2
        return compute_packet_rep_2d(image, wavelet_str, max_lev)


if __name__ == '__main__':

    pairs = []
    pairs.append(read_pair("./data/A_ffhq/00000.png",
                           "./data/B_stylegan/style_gan_ffhq_example0.png"))
    pairs.append(read_pair("./data/A_ffhq/00001.png",
                           "./data/B_stylegan/style_gan_ffhq_example1.png"))
    pairs.append(read_pair("./data/A_ffhq/00002.png",
                           "./data/B_stylegan/style_gan_ffhq_example2.png"))
    pairs.append(read_pair("./data/A_ffhq/00003.png",
                           "./data/B_stylegan/style_gan_ffhq_example3.png"))

    wavelet = 'db2'
    max_lev = 3
    for real, fake in pairs:
        real = torch.from_numpy(np.mean(real, -1).astype(np.float32)).unsqueeze(0)
        fake = torch.from_numpy(np.mean(fake, -1).astype(np.float32)).unsqueeze(0)
        # plt.imshow(np.concatenate([real, fake], axis=1))
        # plt.show()
        real_packets = compute_pytorch_packet_representation_2d(
            real, wavelet_str=wavelet, max_lev=max_lev)
        fake_packets = compute_pytorch_packet_representation_2d(
            fake, wavelet_str=wavelet, max_lev=max_lev)

        # merge_packets = np.concatenate([real_packets, fake_packets], axis=1)
        abs_real_packets = np.abs(real_packets.numpy())
        abs_fake_packets = np.abs(fake_packets.numpy())
        # scaled_packets = abs_packets/np.max(abs_packets)
        # log_scaled_packets = np.log(abs_packets)
        # scaled_packets = np.

        scale_min = np.min([abs_real_packets.min(), abs_fake_packets.min()]) \
            + 2e-4
        scale_max = np.max([abs_real_packets.max(), abs_fake_packets.max()])

        cmap = 'cividis'  # 'cividis'  # 'magma'  #'inferno'  # 'viridis
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_title('real img ' + wavelet + ' packet decomposition')
        ax1.imshow(np.abs(abs_real_packets),
                   norm=colors.LogNorm(vmin=scale_min,
                                       vmax=scale_max),
                   cmap=cmap)
        ax2.set_title('fake img ' + wavelet + ' packet decomposition')
        im = ax2.imshow(np.abs(abs_fake_packets),
                        norm=colors.LogNorm(vmin=scale_min,
                                            vmax=scale_max),
                        cmap=cmap)
        fig.colorbar(im)
        shape = real.shape
        keys = list(product(['a', 'd', 'h', 'v'], repeat=max_lev))
        draw_2d_wp_basis(shape, keys, ax=ax3, label_levels=max_lev)
        ax3.set_title('packet labels')
        plt.show()
