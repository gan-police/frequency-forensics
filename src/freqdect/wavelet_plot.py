import argparse

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pywt._doc_utils import _2d_wp_basis_coords

from .wavelet_math import (
    compute_packet_rep_2d,
    compute_pytorch_packet_representation_2d_tensor,
)


def draw_2d_wp_basis(shape, keys, fmt="k", plot_kwargs={}, ax=None, label_levels=0):
    """Plot a 2D representation of a WaveletPacket2D basis.
    Based on: pywt._doc_utils.draw_2d_wp_basis"""
    coords, centers = _2d_wp_basis_coords(shape, keys)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    for coord in coords:
        ax.plot(coord[0], coord[1], fmt)
    ax.set_axis_off()
    ax.axis("square")
    if label_levels > 0:
        for key, c in centers.items():
            if len(key) <= label_levels:
                ax.text(
                    c[0],
                    c[1],
                    "".join(key),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=6,
                )
    return fig, ax


def read_pair(path_real, path_fake):
    face = cv2.cvtColor(cv2.imread(path_real), cv2.COLOR_BGR2RGB) / 255.0
    fake_face = cv2.cvtColor(cv2.imread(path_fake), cv2.COLOR_BGR2RGB) / 255.0
    return face, fake_face


def compute_packet_rep_img(image, wavelet_str, max_lev):

    if len(image.shape) == 3:
        channels_lst = []
        for channel in range(3):
            channels_lst.append(
                compute_packet_rep_2d(image[:, :, channel], wavelet_str, max_lev)
            )
        return np.stack(channels_lst, axis=-1)
    else:
        assert len(image.shape) == 2
        return compute_packet_rep_2d(image, wavelet_str, max_lev)


def main():
    parser = argparse.ArgumentParser(
        description="Plot wavelet decomposition of real and fake imgs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/",
        help="path of folder containing the data (default: ./data/)",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="A_ffhq",
        help="name of folder with real data (default: A_ffhq)",
    )
    parser.add_argument(
        "--fake-data",
        type=str,
        default="B_stylegan",
        help="name of folder with fake data (default: B_stylegan)",
    )
    args = parser.parse_args()

    print(args)

    pairs = []
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00000.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example0.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00001.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example1.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00002.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example2.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00003.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example3.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00004.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example4.png",
        )
    )
    pairs.append(
        read_pair(
            args.data_dir + args.real_data + "/00005.png",
            args.data_dir + args.fake_data + "/style_gan_ffhq_example5.png",
        )
    )

    wavelet = "db1"
    max_lev = 3
    for real, fake in pairs:
        real = (
            torch.from_numpy(np.mean(real, -1).astype(np.float32)).unsqueeze(0).cuda()
        )
        fake = (
            torch.from_numpy(np.mean(fake, -1).astype(np.float32)).unsqueeze(0).cuda()
        )
        # plt.imshow(np.concatenate([real, fake], axis=1))
        # plt.show()
        real_packets = compute_pytorch_packet_representation_2d_tensor(
            real, wavelet_str=wavelet, max_lev=max_lev
        )
        fake_packets = compute_pytorch_packet_representation_2d_tensor(
            fake, wavelet_str=wavelet, max_lev=max_lev
        )

        real_packets = torch.squeeze(real_packets)
        fake_packets = torch.squeeze(fake_packets)

        # merge_packets = np.concatenate([real_packets, fake_packets], axis=1)
        abs_real_packets = np.abs(real_packets.cpu().numpy())
        abs_fake_packets = np.abs(fake_packets.cpu().numpy())
        # scaled_packets = abs_packets/np.max(abs_packets)
        # log_scaled_packets = np.log(abs_packets)
        # scaled_packets = np.

        scale_min = np.min([abs_real_packets.min(), abs_fake_packets.min()]) + 2e-4
        scale_max = np.max([abs_real_packets.max(), abs_fake_packets.max()])

        cmap = "cividis"  # 'cividis'  # 'magma'  #'inferno'  # 'viridis
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # ax3 = fig.add_subplot(133)
        ax1.set_title("real img " + wavelet + " packet decomposition")
        ax1.imshow(
            abs_real_packets,
            norm=colors.LogNorm(vmin=scale_min, vmax=scale_max),
            cmap=cmap,
        )
        ax2.set_title("fake img " + wavelet + " packet decomposition")
        im = ax2.imshow(
            abs_fake_packets,
            norm=colors.LogNorm(vmin=scale_min, vmax=scale_max),
            cmap=cmap,
        )
        # fig.colorbar(im)
        # shape = real.shape
        # keys = list(product(['a', 'd', 'h', 'v'], repeat=max_lev))
        # draw_2d_wp_basis(shape, keys, ax=ax3, label_levels=max_lev)
        # ax3.set_title('packet labels')
        plt.show()

        plt.semilogy(np.mean(abs_real_packets, 0), label="real")
        plt.semilogy(np.mean(abs_fake_packets, 0), label="fake")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
