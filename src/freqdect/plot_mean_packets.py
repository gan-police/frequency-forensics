"""Source code to visualize mean wavelet packets and their standard deviation for visual inspection."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data_loader import LoadNumpyDataset


def _plot_mean_std(x, mean, std, color, label="", marker="."):
    plt.plot(x, mean, label=label, color=color, marker=marker)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def generate_packet_image_tensor(packet_array: torch.Tensor) -> torch.Tensor:
    """Arrange a packet tensor  as an image for imshow.

    Args:
        packet_array ([torch.Tensor): The [bach_size, packet_no, height, width, channels] packets
    Returns:
        [torch.Tensor]: The image of shape [batch_size, height, width, channels]
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
    return torch.cat(img, dim=1)


def generate_natural_packet_image(packet_array: np.ndarray, degree: int):
    """Arrange a  packet array  as an image for imshow.

    Args:
        packet_array ([np.ndarray): The [packet_no, packet_height, packet_width] packets
        degree (int): The degree of the transformation.
    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """

    def _cat_sector(elements: np.ndarray, level: int, max_level: int):
        element_lst = np.split(elements, 4)
        if level < max_level - 1:
            img0 = _cat_sector(element_lst[0], level + 1, max_level)
            img1 = _cat_sector(element_lst[1], level + 1, max_level)
            img2 = _cat_sector(element_lst[2], level + 1, max_level)
            img3 = _cat_sector(element_lst[3], level + 1, max_level)
            return np.concatenate(
                [
                    np.concatenate([img0, img1], axis=2),
                    np.concatenate([img2, img3], axis=2),
                ],
                1,
            )
        else:
            img = np.concatenate(
                [
                    np.concatenate([element_lst[0], element_lst[1]], axis=2),
                    np.concatenate([element_lst[2], element_lst[3]], axis=2),
                ],
                1,
            )
            return img

    return _cat_sector(packet_array, 0, degree).squeeze()


def generate_frequency_packet_image(packet_array: np.ndarray, degree: int):
    """Create a ready-to-polt image with frequency-order packages.

       Given a packet array in natural order, creat an image which is
       ready to plot in frequency order.

    Args:
        packet_array (np.ndarray): [packet_no, packet_height, packet_width]
            in natural order.
        degree (int): The degree of the packet decomposition.

    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """
    wp_freq_path, wp_natural_path = get_freq_order(degree)

    image = []
    # go through the rows.
    for row_paths in wp_freq_path:
        row = []
        for row_path in row_paths:
            index = wp_natural_path.index(row_path)
            packet = packet_array[index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 0)


def get_freq_order(level: int):
    """Get the frequency order for a given packet decomposition level.

    Adapted from:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py

    The code elements denote the filter application order. The filters
    are named following the pywt convention as:
    a - LL, low-low coefficients
    h - LH, low-high coefficients
    v - HL, high-low coefficients
    d - HH, high-high coefficients
    """
    wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))

    def _get_graycode_order(level, x="a", y="d"):
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _expand_2d_path(path):
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes: dict = {}
    for (row_path, col_path), node in [
        (_expand_2d_path(node), node) for node in wp_natural_path
    ]:
        nodes.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes_list: list = [nodes[path] for path in graycode_order if path in nodes]
    wp_frequency_path = []
    for row in nodes_list:
        wp_frequency_path.append([row[path] for path in graycode_order if path in row])
    return wp_frequency_path, wp_natural_path


def main():
    """Compute mean wavelet packets and the standard deviation for a NumPy dataset."""
    import matplotlib.pyplot as plt

    # raw images - use only the training set.
    train_packet_set = LoadNumpyDataset(
        "/nvme/mwolter/ffhq1024x1024_log_packets_haar_reflect_train"
    )
    # train_packet_set = LoadNumpyDataset(
    #     "/nvme/mwolter/source_data_log_packets_train"
    # )

    fake_labels = [1, 2, 3, 4]

    fake_list = []
    real_list = []
    for img_no in range(train_packet_set.__len__()):
        train_element = train_packet_set.__getitem__(img_no)
        packets = train_element["image"].numpy()
        label = train_element["label"].numpy()
        if label in fake_labels:
            fake_list.append(packets)
        elif label == 0:
            real_list.append(packets)
        else:
            print("skipping label", label)

        if img_no % 500 == 0 and img_no > 0:
            print(img_no, "of", train_packet_set.__len__(), "loaded")
            # break

    fake_array = np.array(fake_list)
    del fake_list
    real_array = np.array(real_list)
    del real_list
    print("train set loaded.", fake_array.shape, real_array.shape)

    # mean image plots
    fake_mean_packet_image = generate_frequency_packet_image(
        np.mean(fake_array, axis=(0, -1)), degree=3
    )
    real_mean_packet_image = generate_frequency_packet_image(
        np.mean(real_array, axis=(0, -1)), degree=3
    )
    # std image plots
    fake_std_packet_image = generate_frequency_packet_image(
        np.std(fake_array, axis=(0, -1)), degree=3
    )
    real_std_packet_image = generate_frequency_packet_image(
        np.std(real_array, axis=(0, -1)), degree=3
    )

    fig = plt.figure(figsize=(8, 6))
    columns = 3
    rows = 2
    plot_count = 1
    cmap = "cividis"  # 'magma'  #'inferno'  # 'viridis

    mean_vmin = np.min((np.min(fake_mean_packet_image), np.min(real_mean_packet_image)))
    mean_vmax = np.max((np.max(fake_mean_packet_image), np.max(real_mean_packet_image)))
    std_vmin = np.min((np.min(fake_std_packet_image), np.min(real_std_packet_image)))
    std_vmax = np.max((np.max(fake_std_packet_image), np.max(real_std_packet_image)))

    def _plot_image(image, title, vmax=None, vmin=None):
        fig.add_subplot(rows, columns, plot_count)
        plt.imshow(image, cmap=cmap, vmax=vmax, vmin=vmin)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title)
        plt.colorbar()

    _plot_image(fake_mean_packet_image, "gan mean packets", mean_vmax, mean_vmin)
    plot_count += 1
    _plot_image(real_mean_packet_image, "data-set mean packets", mean_vmax, mean_vmin)
    plot_count += 1
    _plot_image(
        np.abs(fake_mean_packet_image - real_mean_packet_image),
        "absolute mean difference",
    )
    plot_count += 1
    _plot_image(fake_std_packet_image, "gan std packets", std_vmax, std_vmin)
    plot_count += 1
    _plot_image(real_std_packet_image, "data-set std packets", std_vmax, std_vmin)
    plot_count += 1
    _plot_image(
        np.abs(fake_std_packet_image - real_std_packet_image), "absolute std difference"
    )
    plot_count += 1

    plt.savefig("plot.png")
    if 0:
        import tikzplotlib

        tikzplotlib.save("ffhq_style_packet_mean_std_plot.tex", standalone=True)
    plt.show()
    print("first plot done")

    # mean packet plots
    style_gan_mean = np.mean(fake_array, axis=(0, 2, 3, 4))
    style_gan_std = np.std(fake_array, axis=(0, 2, 3, 4))
    ffhq_mean = np.mean(real_array, axis=(0, 2, 3, 4))
    ffhq_std = np.std(real_array, axis=(0, 2, 3, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.array(range(len(style_gan_mean)))
    wp_keys = list(product(["a", "h", "v", "d"], repeat=3))
    wp_labels = ["".join(key) for key in wp_keys]
    _plot_mean_std(x, ffhq_mean, ffhq_std, colors[0], "real data")
    _plot_mean_std(x, style_gan_mean, style_gan_std, colors[1], "gan")
    plt.legend()
    plt.xlabel("filter")
    plt.xticks(x, labels=wp_labels)
    plt.xticks(rotation=80)
    plt.ylabel("mean absolute coefficient magnitude")
    plt.title("Mean absolute coefficient comparison real data-GAN")

    plt.savefig("plot2.png")
    if 0:
        import tikzplotlib

        tikzplotlib.save("absolute_coeff_comparison.tex", standalone=True)
    plt.show()


if __name__ == "__main__":
    main()
