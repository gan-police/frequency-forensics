"""Code to plot training mean accuracy as well as the standard deviation."""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def stack_list(dict_list, key: str):
    """Extract time series data from a logfile-list.

    Args:
        dict_list (list): A list as stored by train_classifier.py
        key (str): The key for a logfile entry.

    Returns:
        tuple: A tuple of a step and accuracy numpy array.
    """
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


def _get_steps_mean_std(step_lst, cost_lst):
    mean = np.mean(cost_lst, axis=0)
    std = np.std(cost_lst, axis=0)
    return step_lst[0, :], mean, std


def get_plot_tuple(dict_list, key: str):
    """Extract time series data from a logfile-list.

    Args:
        dict_list (list): A list as stored by train_classifier.py
        key (str): The key for a logfile entry.

    Returns:
        tuple: A tuple of a step and mean accuracy and standard deviation.
    """
    steps, loss = stack_list(dict_list, key)
    steps, mean, std = _get_steps_mean_std(steps, loss)
    return steps, mean, std


def _plot_mean_std(axs, steps, mean, std, color, label="", marker="."):
    axs.plot(steps, mean, label=label, color=color, marker=marker)
    axs.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def get_test_acc_mean_std_max(dict_list: list, key: str):
    """Compute the mean test accuracy and standard deviation over multiple runs.

    Args:
        dict_list (list): A list of dicts as stored by train_classifier.py
        key (str): The dictionary key we are interested in.

    Returns:
        tuple: The mean, standard deviation and max in that order.
    """
    test_accs = []
    for experiment_dict in dict_list:
        test_accs.append(experiment_dict[key])
    return np.mean(test_accs), np.std(test_accs), np.max(test_accs)


def _plot_on_ax(
    dataset: str,
    param_str: str,
    axs: Axes,
    logpacket_logs,
    packet_logs,
    raw_logs,
    epochs: int = None,
    batch_size: int = None,
    ylabel: str = None,
    ylim: float = None,
    title: str = None,
    place_legend: bool = False,
):
    # convert logs to ndarrays to allow better indexing
    if raw_logs:
        raw_logs = np.array(raw_logs)
    if packet_logs:
        packet_logs = np.array(packet_logs)
    if logpacket_logs:
        logpacket_logs = np.array(logpacket_logs)

    log_names = ["raw", "packets", "log_packets"]

    # filter out all log entries that do not match the specified epoch number
    if epochs is not None:
        if raw_logs is not None:
            indices_raw = [vars(run["args"])["epochs"] == epochs for run in raw_logs]
            raw_logs = raw_logs[indices_raw]
        if packet_logs is not None:
            indices_packets = [
                vars(run["args"])["epochs"] == epochs for run in packet_logs
            ]
            packet_logs = packet_logs[indices_packets]
        if logpacket_logs is not None:
            indices_logpackets = [
                vars(run["args"])["epochs"] == epochs for run in logpacket_logs
            ]
            logpacket_logs = logpacket_logs[indices_logpackets]

        for logs, logs_name in zip([raw_logs, packet_logs, logpacket_logs], log_names):
            if logs is not None and logs.size == 0:
                print(f"No runs found for {epochs} epochs for {logs_name}")

    # filter out all log entries that do not match the specified batch_size number
    if batch_size is not None:
        if raw_logs is not None:
            indices_raw = [
                vars(run["args"])["batch_size"] == epochs for run in raw_logs
            ]
            raw_logs = raw_logs[indices_raw]
        if packet_logs is not None:
            indices_packets = [
                vars(run["args"])["batch_size"] == epochs for run in packet_logs
            ]
            packet_logs = packet_logs[indices_packets]
        if logpacket_logs is not None:
            indices_logpackets = [
                vars(run["args"])["batch_size"] == epochs for run in logpacket_logs
            ]
            logpacket_logs = logpacket_logs[indices_logpackets]

        for logs, logs_name in zip([raw_logs, packet_logs, logpacket_logs], log_names):
            if logs is not None and logs.size == 0:
                print(f"No runs found for {batch_size} epochs for {logs_name}")

    print(f"{dataset} {param_str}:")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def print_results(name, logs, logs_mean, logs_std, logs_max):
        """Print the max, mean and std of the accuracy of the runs on one feature."""
        print(f"{name} ({len(logs)} runs):")
        print(
            f"\t{name} seeds:",
            ", ".join([str(vars(run["args"])["seed"]) for run in logs]),
        )
        print(
            f"\t\tmax: {logs_max * 100:.2f}%\n\t\tmean: {logs_mean * 100:.2f}%\n\t\tstd: {logs_std * 100:.2f}"
        )

    def _process_logs(name, logs, idx):
        steps, mean, std = get_plot_tuple(logs, "val_acc")
        _plot_mean_std(
            axs, steps, mean, std, color=colors[idx], label=f"{name} validation acc"
        )
        t_mean, t_std, t_max = get_test_acc_mean_std_max(logs, "test_acc")
        print_results(name, logs, t_mean, t_std, t_max)

        axs.errorbar(
            logs[0]["train_acc"][-1][0],
            t_mean,
            t_std,
            color=colors[3 + idx],
            label=f"{name} test acc",
            marker="_",
        )

    for idx, (logs, logs_name) in enumerate(
        zip([raw_logs, packet_logs, logpacket_logs], log_names)
    ):
        if logs is not None:
            _process_logs(logs_name, logs, idx)

    axs.set_xlabel("training steps")
    if ylabel is not None:
        axs.set_ylabel(ylabel)

    if ylim is not None:
        axs.set_ylim(top=ylim)

    if title is not None:
        axs.set_title(title)
    else:
        axs.set_title(f"{dataset}-GAN")

    if place_legend:
        axs.legend()


def export_plots(args, output_prefix: str):
    """Export the plot as png or tikz plot.

    Shows the plot, if not specified otherwise in the cmd line args.

    Args:
        args: The cmd line args settings.
        output_prefix (str): A prefix, with which the file names of the exported plots start.
    """
    output_str = ""
    if args.wavelet:
        output_str += f"_{args.wavelet}"
    if args.mode:
        output_str += f"_{args.mode}"
    if args.learning_rate:
        output_str += f"_{args.learning_rate}"
    if args.epochs:
        output_str += f"_{args.epochs}e"
    output_str += f"_{args.model}"

    if args.png:
        print(f"saving {output_prefix}{output_str}_accuracy.png")
        plt.savefig(f"{output_prefix}{output_str}_accuracy.png")
    if args.tikz:
        import tikzplotlib

        print(f"saving {output_prefix}{output_str}_accuracy.tex")
        tikzplotlib.save(f"{output_prefix}{output_str}_accuracy.tex", standalone=True)
    if not args.hide:
        plt.show()


def skip_every_second_val_acc(logs):
    """Half the validation accuracy resolution by skipping every second validation accuracy entry.

    If the interval between the validation accuracy measurements is too small, the resulting plot is too loaded.
    In this case, this function is useful.

    Args:
        logs: The log of the runs, from which every second validation accuracy measurement is skipped.
    """
    for run in logs:
        run["val_acc"] = run["val_acc"][1::2]


def plot_shared(args):
    """Plot the validation and test accuracy.

    Both LSUN and CelebA are shown side by side for better comparision.
    """
    logpacket_logs_lsun = pickle.load(
        open(f"{args.prefix_lsun}_logpackets_{args.model}.pkl", "rb")
    )
    packet_logs_lsun = pickle.load(
        open(f"{args.prefix_lsun}_packets_{args.model}.pkl", "rb")
    )
    raw_logs_lsun = pickle.load(open(f"{args.prefix_lsun}_raw_{args.model}.pkl", "rb"))
    logpacket_logs_celeba = pickle.load(
        open(f"{args.prefix_celeba}_logpackets_{args.model}.pkl", "rb")
    )
    packet_logs_celeba = pickle.load(
        open(f"{args.prefix_celeba}_packets_{args.model}.pkl", "rb")
    )
    raw_logs_celeba = pickle.load(
        open(f"{args.prefix_celeba}_raw_{args.model}.pkl", "rb")
    )

    if args.skip_val_acc_indices is not None:
        log_list = [
            raw_logs_celeba,
            packet_logs_celeba,
            logpacket_logs_celeba,
            raw_logs_lsun,
            packet_logs_lsun,
            logpacket_logs_lsun,
        ]
        for idx in args.skip_val_acc_indices:
            skip_every_second_val_acc(log_list[idx])

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

    _plot_on_ax(
        dataset="CelebA",
        model=args.model,
        axs=ax2,
        logpacket_logs=logpacket_logs_celeba,
        packet_logs=packet_logs_celeba,
        raw_logs=raw_logs_celeba,
        epochs=args.epochs[1],
        batch_size=args.batch_size[1],
        ylim=args.ylim,
    )

    _plot_on_ax(
        dataset="LSUN",
        model=args.model,
        axs=ax1,
        logpacket_logs=logpacket_logs_lsun,
        packet_logs=packet_logs_lsun,
        raw_logs=raw_logs_lsun,
        epochs=args.epochs[0],
        batch_size=args.batch_size[0],
        ylim=args.ylim,
        ylabel="accuracy",
    )

    plt.suptitle("source identification")
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.30))
    plt.tight_layout()

    export_plots(args, output_prefix="lsun_celeba")


def plot_single(args):
    """Plot the validation and test accuracy for one data set."""
    suffix_str_packets = ""
    suffix_str_raw = ""
    params_str = f"{args.model}"
    if args.wavelet:
        suffix_str_packets += f"_{args.wavelet}"
        params_str += f" {args.wavelet}"
    if args.mode:
        suffix_str_packets += f"_{args.mode}"
        params_str += f" {args.mode}"
    if args.learning_rate:
        suffix_str_packets += f"_{args.learning_rate}"
        suffix_str_raw += f"_{args.learning_rate}"
        params_str += f" {args.learning_rate}"
    if args.epochs:
        suffix_str_packets += f"_{args.epochs}e"
        suffix_str_raw += f"_{args.epochs}e"
        params_str += f" {args.epochs}e"
    suffix_str_packets += f"_{args.model}"
    suffix_str_raw += f"_{args.model}"

    try:
        logpacket_logs = pickle.load(
            open(f"{args.prefix}_log_packets{suffix_str_packets}.pkl", "rb")
        )
    except FileNotFoundError:
        print(f"{args.prefix}_log_packets{suffix_str_packets}.pkl not found!")
        logpacket_logs = None
    try:
        packet_logs = pickle.load(
            open(f"{args.prefix}_packets{suffix_str_packets}.pkl", "rb")
        )
    except FileNotFoundError:
        print(f"{args.prefix}_packets{suffix_str_packets}.pkl not found!")
        packet_logs = None
    try:
        raw_logs = pickle.load(open(f"{args.prefix}_raw{suffix_str_raw}.pkl", "rb"))
    except FileNotFoundError:
        raw_logs = None
        print(f"{args.prefix}_raw{suffix_str_raw}.pkl not found!")

    if not any([logpacket_logs, packet_logs, raw_logs]):
        raise ValueError("Not log files found!")

    if args.skip_val_acc_indices is not None:
        log_list = [raw_logs, packet_logs, logpacket_logs]
        for idx in args.skip_val_acc_indices:
            if log_list[idx]:
                skip_every_second_val_acc(log_list[idx])

    _plot_on_ax(
        dataset=args.dataset,
        param_str=params_str,
        axs=plt.gca(),
        logpacket_logs=logpacket_logs,
        packet_logs=packet_logs,
        raw_logs=raw_logs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ylabel="accuracy",
        ylim=args.ylim,
        place_legend=True,
        title=f"{args.dataset} {params_str} binary classification",
    )

    export_plots(args, output_prefix=args.dataset.lower())


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot validation accuracy")

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("model", choices=["regression", "cnn"])
    parent_parser.add_argument(
        "-p", "--png", action="store_true", help="save the plot as a png"
    )
    parent_parser.add_argument(
        "-t", "--tikz", action="store_true", help="export a tikz version of the plot"
    )
    parent_parser.add_argument(
        "--hide", action="store_true", help="do not show the plot"
    )
    parent_parser.add_argument(
        "--skip-val-acc-indices",
        nargs="*",
        type=int,
        default=None,
        help="indices of the logs, for which every second validation accuracy value should be "
        "skipped (starting at 0). The order of lists is [raw, packets, logpackets] (and "
        "[celeba, lsun] in the shared case), e.g. for lsun packets the index would be 1 "
        "(or 4 in the shared case).",
    )
    parent_parser.add_argument(
        "--ylim", type=float, default=None, help="Maximal value of the y axis"
    )
    parent_parser.add_argument("--wavelet", type=str, default=None, help="Wavelet used")
    parent_parser.add_argument(
        "--mode", type=str, default=None, help="Boundary mode used"
    )
    parent_parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate used"
    )

    subparsers = parser.add_subparsers(required=True)

    # create subparser for plotting a shared plot for LSUN/CelebA
    parser_shared = subparsers.add_parser("shared", parents=[parent_parser])
    parser_shared.add_argument(
        "--epochs",
        nargs=2,
        metavar=("LSUN_EPOCHS", "CELEBA_EPOCHS"),
        type=int,
        default=[None, None],
        help="Filter the logs for only these numbers of epochs",
    )
    parser_shared.add_argument(
        "--batch-size",
        nargs=2,
        metavar=("LSUN_BATCH_SIZE", "CELEBA_BATCH_SIZE"),
        type=int,
        default=[None, None],
        help="Filter the logs for only these batch sizes",
    )
    parser_shared.add_argument(
        "--prefix-lsun",
        type=str,
        default="./log/lsun_bedroom_200k_png",
        help="shared file path prefix of the log files (default: ./log/lsun_bedroom_200k_png)",
    )
    parser_shared.add_argument(
        "--prefix-celeba",
        default="./log/celeba_align_png_cropped",
        help="shared file path prefix of the log files (default: ./log/celeba_align_png_cropped)",
    )
    parser_shared.set_defaults(func=plot_shared)

    # create subparser for plotting either LSUN or CelebA
    parser_lsun = subparsers.add_parser("lsun", parents=[parent_parser])
    parser_lsun.add_argument(
        "--prefix",
        default="./log/lsun_bedroom_200k_png",
        help="shared file path prefix of the log files (default: ./log/lsun_bedroom_200k_png)",
    )
    parser_lsun.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Filter the logs for only this number of epochs",
    )
    parser_lsun.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Filter the logs for only this batch size",
    )
    parser_lsun.set_defaults(func=plot_single)
    parser_lsun.set_defaults(dataset="LSUN")

    parser_celeba = subparsers.add_parser("celeba", parents=[parent_parser])
    parser_celeba.add_argument(
        "--prefix",
        default="./log/celeba_align_png_cropped",
        help="shared file path prefix of the log files (default: ./log/celeba_align_png_cropped)",
    )
    parser_celeba.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Filter the logs for only this number of epochs",
    )
    parser_celeba.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Filter the logs for only this batch size",
    )
    parser_celeba.set_defaults(func=plot_single)
    parser_celeba.set_defaults(dataset="CelebA")

    parser_other = subparsers.add_parser("other", parents=[parent_parser])
    parser_other.add_argument(
        "--dataset", type=str, default="other_dataset", help="Name of the dataset"
    )
    parser_other.add_argument(
        "--prefix",
        type=str,
        default="./log/data",
        help="shared file path prefix of the log files",
    )
    parser_other.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Filter the logs for only this number of epochs",
    )
    parser_other.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Filter the logs for only this batch size",
    )
    parser_other.set_defaults(func=plot_single)

    return parser.parse_args()


def main(args):
    """Plot the accuracy results, as specified in the cmd line args."""
    args.func(args)


if __name__ == "__main__":
    main(_parse_args())
