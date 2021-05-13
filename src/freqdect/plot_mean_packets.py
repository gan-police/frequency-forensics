from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from data_loader import LoadNumpyDataset


def _plot_mean_std(x, mean, std, color, label="", marker="."):
    plt.plot(x, mean, label=label, color=color, marker=marker)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def _generate_packet_image(packet_array):
    """ Arrange a  packet array  as an image for imshow.

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


def main():
    import matplotlib.pyplot as plt

    # raw images - use only the training set.
    train_packet_set = LoadNumpyDataset("./data/source_data_packets_train")

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
            raise ValueError

    print("train set loaded.")
    style_gan_array = np.array(style_gan_list)
    del style_gan_list
    ffhq_array = np.array(ffhq_list)
    del ffhq_list

    # mean image plots
    gan_mean_packet_image = _generate_packet_image(np.mean(style_gan_array, axis=(0, -1)))
    ffhq_mean_packet_image = _generate_packet_image(np.mean(ffhq_array, axis=(0, -1)))

    fig = plt.figure(figsize=(8, 6))
    columns = 3
    rows = 1
    plot_count = 1
    cmap = 'cividis'  # 'magma'  #'inferno'  # 'viridis

    def plot_image(image, title):
        fig.add_subplot(rows, columns, plot_count)
        plt.imshow(image, cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title)

    plot_image(gan_mean_packet_image, 'gan mean packets')
    plot_count += 1
    plot_image(ffhq_mean_packet_image, 'ffhq mean packets')
    plot_count += 1
    plot_image(np.abs(gan_mean_packet_image - ffhq_mean_packet_image),
               'absolute difference')
    plot_count += 1
    plt.show()

    # mean packet plots
    style_gan_mean = np.mean(style_gan_array, axis=(0, 2, 3, 4))
    style_gan_std = np.std(style_gan_array, axis=(0, 2, 3, 4))

    ffhq_mean = np.mean(ffhq_array, axis=(0, 2, 3, 4))
    ffhq_std = np.std(ffhq_array, axis=(0, 2, 3, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.array(range(len(style_gan_mean)))
    wp_keys = list(product(["a", "d", "h", "v"], repeat=3))
    wp_labels = ["".join(key) for key in wp_keys]
    _plot_mean_std(x, ffhq_mean, ffhq_std, colors[0], "ffhq")
    _plot_mean_std(x, style_gan_mean, style_gan_std, colors[1], "style gan")
    plt.legend()
    plt.xlabel("filter")
    plt.xticks(x, labels=wp_labels)
    plt.xticks(rotation=80)
    plt.ylabel("mean absolute coefficient magnitude")
    plt.title("Mean absolute coefficient comparison FFHQ-StyleGAN")

    if 0:
        import tikzplotlib

        tikzplotlib.save("mean_absolute_coeff_comparison.tex", standalone=True)
    else:
        plt.show()


if __name__ == "__main__":
    main()
