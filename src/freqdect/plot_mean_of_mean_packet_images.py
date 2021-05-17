import matplotlib.pyplot as plt
import numpy as np
from data_loader import LoadNumpyDataset
from plot_mean_packets import generate_packet_image


def _plot_mean_std(x, mean, std, color, label="", marker="."):
    plt.plot(x, mean, label=label, color=color, marker=marker)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def main():
    # https://en.wikipedia.org/wiki/Grand_mean
    # raw images - use only the training set.
    train_packet_set = LoadNumpyDataset("./data/ffhq_stylegan_large_packets_train_2")

    style_gan_list = []
    ffhq_list = []
    gan_mean_packet_lst = []
    ffhq_mean_packet_lst = []
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

        if len(style_gan_list) >= train_packet_set.__len__() // 8:
            print('pool gan', 'list lengths', len(style_gan_list), len(ffhq_list))
            gan_mean_packet_lst.append(np.mean(np.array(style_gan_list), axis=(0, -1)))
            style_gan_list = []

        if len(ffhq_list) >= train_packet_set.__len__() // 8:
            print('pool ffhq', 'list lengths', len(style_gan_list), len(ffhq_list))
            ffhq_mean_packet_lst.append(np.mean(np.array(ffhq_list), axis=(0, -1)))
            ffhq_list = []

    print("train set loaded.")
    grand_gan_mean_packet = np.mean(np.stack(gan_mean_packet_lst, axis=0), axis=0)
    grand_ffhq_mean_packet = np.mean(np.stack(ffhq_mean_packet_lst, axis=0), axis=0)

    # mean image plots
    gan_mean_packet_image = generate_packet_image(grand_gan_mean_packet)
    ffhq_mean_packet_image = generate_packet_image(grand_ffhq_mean_packet)

    fig = plt.figure(figsize=(8, 6))
    columns = 3
    rows = 1
    plot_count = 1
    cmap = 'cividis'  # 'magma'  #'inferno'  # 'viridis
    vmin = np.min((np.min(gan_mean_packet_image), np.min(ffhq_mean_packet_image))) + 2e-4
    vmax = np.max((np.max(gan_mean_packet_image), np.max(ffhq_mean_packet_image)))

    def plot_image(image, title, vmin=None, vmax=None):
        fig.add_subplot(rows, columns, plot_count)
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title)
        plt.colorbar()

    plot_image(gan_mean_packet_image, 'gan mean packets', vmin, vmax)
    plot_count += 1
    plot_image(ffhq_mean_packet_image, 'ffhq mean packets', vmin, vmax)
    plot_count += 1
    plot_image(np.abs(gan_mean_packet_image - ffhq_mean_packet_image),
               'absolute mean difference')
    plt.show()

    if 0:
        import tikzplotlib

        tikzplotlib.save("mean_absolute_coeff_comparison.tex", standalone=True)
    else:
        plt.show()
    print('done')


if __name__ == "__main__":
    main()
