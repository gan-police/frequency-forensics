import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


def stack_list(dict_list, key: str):
    step_lst = []
    acc_lst = []
    for current_dictionary in dict_list:
        if len(current_dictionary[key][0]) == 2:
            steps, acc = zip(*current_dictionary[key])
        elif len(current_dictionary[key][0]) == 3:
            steps, epochs, acc = zip(*current_dictionary[key])
        step_lst.append(steps)
        acc_lst.append(acc)
    return np.stack(step_lst), np.stack(acc_lst)


def get_steps_mean_std(step_lst, cost_lst):
    mean = np.mean(cost_lst, axis=0)
    std = np.std(cost_lst, axis=0)
    return step_lst[0, :], mean, std


def get_plot_tuple(dict_list, key: str):
    steps, loss = stack_list(dict_list, key)
    steps, mean, std = get_steps_mean_std(steps, loss)
    return steps, mean, std


def plot_mean_std(steps, mean, std, color, label="", marker="."):
    plt.plot(steps, mean, label=label, color=color, marker=marker)
    plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def get_test_acc_mean_std_max(dict_list: dict, key: str):
    test_accs = []
    for experiment_dict in dict_list:
        test_accs.append(experiment_dict[key])
    return np.mean(test_accs), np.std(test_accs), np.max(test_accs)


def main(args):
    packet_logs = pickle.load(open(args.PACKETFILE, "rb"))
    raw_logs = pickle.load(open(args.PIXELFILE, "rb"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    steps, mean, std = get_plot_tuple(raw_logs, "train_acc")
    steps, mean, std = get_plot_tuple(raw_logs, "val_acc")
    plot_mean_std(steps, mean, std, color=colors[0], label="raw validation acc")

    steps, mean, std = get_plot_tuple(packet_logs, "val_acc")
    plot_mean_std(steps, mean, std, color=colors[1], label="packet validation acc")

    pt_mean, pt_std, pt_max = get_test_acc_mean_std_max(packet_logs, "test_acc")
    rt_mean, rt_std, rt_max = get_test_acc_mean_std_max(raw_logs, "test_acc")
    print("packet_mean", pt_mean, "packet_std", pt_std, "packet_max", pt_max)
    print("raw_mean", rt_mean, "raw_std", rt_std, "raw_max", rt_max)
    plt.errorbar(
        steps[-1], pt_mean, pt_std, color=colors[2], label="packet test acc", marker="x"
    )
    plt.errorbar(
        steps[-1], rt_mean, rt_std, color=colors[3], label="raw test acc", marker="x"
    )

    plt.ylabel("mean accuracy")
    plt.xlabel("training steps")
    plt.title("Accuracy Celeba-GAN source identification")
    plt.legend()
    if args.tikz:
        import tikzplotlib

        tikzplotlib.save("celeba_source_identification.tex", standalone=True)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    # "./log/celeba_align_png_cropped_packets_regression.pkl"
    parser.add_argument("PACKETFILE", help="packet log to plot.", type=str)
    # "./log/celeba_align_png_cropped_raw_regression.pkl"
    parser.add_argument("PIXELFILE", help="packet log to plot.", type=str)
    parser.add_argument("TITLE", help="the plot title", type=str)
    parser.add_argument(
        "--tikz", action="store_true", help="use tikz output imstead of plt.show()"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
