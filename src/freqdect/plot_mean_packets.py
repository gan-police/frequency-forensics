""" Source code to visualize mean wavelet packets and their
    standard deviation for visual inspection. """

import matplotlib.pyplot as plt
from itertools import product
import torch
import numpy as np

from data_loader import LoadNumpyDataset


def _plot_mean_std(x, mean, std, color, label="", marker="."):
    plt.plot(x, mean, label=label, color=color, marker=marker)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def generate_packet_image(packet_array: np.array):
    """Arrange a  packet array  as an image for imshow.
    Args:
        packet_array ([np.array): The [packet_no, height, width] packets
    Returns:
        [np.array]: The image of shape [height, width]
    """
    packet_count = packet_array.shape[0]
    count = 0
    img_rows = None
    img = []
    for node_no in range(packet_count):
        packet = packet_array[node_no]
        if img_rows is not None:
            img_rows = np.concatenate([img_rows, packet], axis=1)
        else:
            img_rows = packet
        count += 1
        if count >= np.sqrt(packet_count):
            count = 0
            img.append(img_rows)
            img_rows = None
    img = np.concatenate(img, axis=0)
    return img


def generate_packet_image_tensor(packet_array: torch.tensor):
    """Arrange a  packet tensor  as an image for imshow.
    Args:
        packet_array ([torch.tensor): The [bach_size, packet_no, height, width, channels] packets
    Returns:
        [torch.tensor]: The image of shape [batch_size, height, width, channels]
    """
    packet_count = packet_array.shape[1]
    count = 0
    img_rows = None
    img = []
    for node_no in range(packet_count):
        packet = packet_array[:, node_no]
        if img_rows is not None:
            img_rows = torch.cat([img_rows, packet], axis=2)
        else:
            img_rows = packet
        count += 1
        if count >= np.sqrt(packet_count):
            count = 0
            img.append(img_rows)
            img_rows = None
    img = torch.cat(img, axis=1)
    return img


def main():
    """ Compute mean wavelet packets and the standard deviation for a NumPy 
        dataset. """

    import matplotlib.pyplot as plt

    # raw images - use only the training set.
    # train_packet_set = LoadNumpyDataset("/home/ndv/projects/wavelets/frequency-forensics_felix/data/lsun_bedroom_200k_png_baseline_logpackets_train/")
    train_packet_set = LoadNumpyDataset(
        "/home/ndv/projects/wavelets/frequency-forensics_felix/data/celeba_align_png_cropped_baselines_logpackets_train/"
    )

    style_gan_list = []
    ffhq_list = []
    for img_no in range(train_packet_set.__len__()):
        train_element = train_packet_set.__getitem__(img_no)
        packets = train_element["image"].numpy()
        label = train_element["label"].numpy()
        if label == 1:
            style_gan_list.append(packets)
        elif label == 0:
            ffhq_list.append(packets)
        else:
            print("skipping label", label)

        if img_no % 500 == 0 and img_no > 0:
            print(img_no, "of", train_packet_set.__len__(), "loaded")
            # break

    style_gan_array = np.array(style_gan_list)
    del style_gan_list
    ffhq_array = np.array(ffhq_list)
    del ffhq_list
    print("train set loaded.", style_gan_array.shape, ffhq_array.shape)

    # mean image plots
    gan_mean_packet_image = generate_packet_image(
        np.mean(style_gan_array, axis=(0, -1))
    )
    ffhq_mean_packet_image = generate_packet_image(np.mean(ffhq_array, axis=(0, -1)))
    # std image plots
    gan_std_packet_image = generate_packet_image(np.std(style_gan_array, axis=(0, -1)))
    ffhq_std_packet_image = generate_packet_image(np.std(ffhq_array, axis=(0, -1)))

    fig = plt.figure(figsize=(8, 6))
    columns = 3
    rows = 2
    plot_count = 1
    cmap = "cividis"  # 'magma'  #'inferno'  # 'viridis

    mean_vmin = np.min((np.min(gan_mean_packet_image), np.min(ffhq_mean_packet_image)))
    mean_vmax = np.max((np.max(gan_mean_packet_image), np.max(ffhq_mean_packet_image)))
    std_vmin = np.min((np.min(gan_std_packet_image), np.min(ffhq_std_packet_image)))
    std_vmax = np.max((np.max(gan_std_packet_image), np.max(ffhq_std_packet_image)))

    def _plot_image(image, title, vmax=None, vmin=None):
        fig.add_subplot(rows, columns, plot_count)
        plt.imshow(image, cmap=cmap, vmax=vmax, vmin=vmin)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title)
        plt.colorbar()

    _plot_image(gan_mean_packet_image, "gan mean packets", mean_vmax, mean_vmin)
    plot_count += 1
    _plot_image(ffhq_mean_packet_image, "data-set mean packets", mean_vmax, mean_vmin)
    plot_count += 1
    _plot_image(
        np.abs(gan_mean_packet_image - ffhq_mean_packet_image),
        "absolute mean difference",
    )
    plot_count += 1
    _plot_image(gan_std_packet_image, "gan std packets", std_vmax, std_vmin)
    plot_count += 1
    _plot_image(ffhq_std_packet_image, "data-set std packets", std_vmax, std_vmin)
    plot_count += 1
    _plot_image(
        np.abs(gan_std_packet_image - ffhq_std_packet_image), "absolute std difference"
    )
    plot_count += 1

    if 1:
        import tikzplotlib

        tikzplotlib.save("celeba_packet_mean_std_plot.tex", standalone=True)
    plt.show()
    print("first plot done")

    # mean packet plots
    style_gan_mean = np.mean(style_gan_array, axis=(0, 2, 3, 4))
    style_gan_std = np.std(style_gan_array, axis=(0, 2, 3, 4))
    ffhq_mean = np.mean(ffhq_array, axis=(0, 2, 3, 4))
    ffhq_std = np.std(ffhq_array, axis=(0, 2, 3, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.array(range(len(style_gan_mean)))
    wp_keys = list(product(["a", "d", "h", "v"], repeat=3))
    wp_labels = ["".join(key) for key in wp_keys]
    _plot_mean_std(x, ffhq_mean, ffhq_std, colors[0], "real data")
    _plot_mean_std(x, style_gan_mean, style_gan_std, colors[1], "gan")
    plt.legend()
    plt.xlabel("filter")
    plt.xticks(x, labels=wp_labels)
    plt.xticks(rotation=80)
    plt.ylabel("mean absolute coefficient magnitude")
    plt.title("Mean absolute coefficient comparison real data-GAN")

    if 1:
        import tikzplotlib

        tikzplotlib.save("celeba_mean_absolute_coeff_comparison.tex", standalone=True)
    plt.show()
    print("done")


if __name__ == "__main__":
    main()
