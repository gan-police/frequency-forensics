"""Code to plot training mean accuracy as well as the standard deviation."""
import argparse
import pickle

import matplotlib.pyplot as plt

from .plot_accuracy_results import (
    get_plot_tuple,
    get_test_acc_mean_std_max,
)


def plot_mean_std(steps, mean, std, color, label="", marker="."):
    """Plot means and standard deviations with shaded areas."""
    plt.plot(steps, mean, label=label, color=color, marker=marker)
    plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def _parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Simply plot validation accuracy")
    parser.add_argument("prefix_one", type=str)
    parser.add_argument("prefix_two", type=str)
    return parser.parse_args()


def main(args):
    """Plot two experiments."""
    print(args.prefix_one)
    print(args.prefix_two)
    first_logs = pickle.load(open(f"./log/{args.prefix_one}.pkl", "rb"))
    second_logs = pickle.load(open(f"./log/{args.prefix_two}.pkl", "rb"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    steps, mean, std = get_plot_tuple(second_logs, "train_acc")
    steps, mean, std = get_plot_tuple(second_logs, "val_acc")
    plot_mean_std(steps, mean, std, color=colors[0], label="second validation acc")

    steps, mean, std = get_plot_tuple(first_logs, "val_acc")
    plot_mean_std(steps, mean, std, color=colors[1], label="first validation acc")

    pt_mean, pt_std, pt_max = get_test_acc_mean_std_max(first_logs, "test_acc")
    rt_mean, rt_std, rt_max = get_test_acc_mean_std_max(second_logs, "test_acc")
    print("first_mean", pt_mean, "first_std", pt_std, "first_max", pt_max)
    print("second_mean", rt_mean, "second_std", rt_std, "second_max", rt_max)
    plt.errorbar(
        steps[-1], pt_mean, pt_std, color=colors[2], label="first test acc", marker="x"
    )
    plt.errorbar(
        steps[-1], rt_mean, rt_std, color=colors[3], label="second test acc", marker="x"
    )

    plt.ylabel("mean accuracy")
    plt.xlabel("training steps")
    plt.title("Accuracy source identification")
    plt.legend()
    if 1:
        import tikzplotlib as tikz

        tikz.save("ffhq_style_style2.tex", standalone=True)
    plt.show()
    print("done")


if __name__ == "__main__":
    main(_parse_args())
