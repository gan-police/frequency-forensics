"""
Basline code as found at:
https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/baselines/baselines.py
"""

import argparse
from multiprocessing import cpu_count
from pathlib import Path
from typing import Type, Dict

import numpy as np

from .classifier import Classifier, read_dataset
from .eigenface import PCAClassifier
from .knn import KNNClassifier
from .prnu import PRNUClassifier

CLASSIFIER_CLS: Dict[str, Type[Classifier]] = {
    "prnu": PRNUClassifier,
    "eigenfaces": PCAClassifier,
    "knn": KNNClassifier,
}


def main(
    command,
    baseline,
    datasets,
    datasets_dir,
    output_dir,
    n_jobs,
    classifier_name,
    normalize,
    calc_normalization,
    **classifier_args,
):
    """Run baselines."""
    print("[+] ARGUMENTS")
    print(f"    -> command      @ {command}")
    print(f"    -> baseline     @ {baseline}")
    if command == "train":
        print(
            "       * ",
            "\n       * ".join([f"{k} = {v}" for k, v in classifier_args.items()]),
            sep="",
        )
    if command == "test":
        print("       * ", classifier_name)
    print(f"    -> n_jobs       @ {n_jobs}")
    print(f"    -> datasets     @ {len(datasets)}")
    print("       * ", "\n       * ".join(list(datasets)), sep="")
    print(f"    -> datasets_dir @ {datasets_dir}")
    assert datasets_dir.is_dir()
    print(f"    -> output_dir   @ {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # select classifier class of given baseline
    classifier_cls = CLASSIFIER_CLS[baseline]

    # grid search
    if command == "grid_search":
        print("\n[+] GRID SEARCH")
        best_results = {}
        for dataset_name in datasets:
            print(f"\n{dataset_name.upper()}")

            if normalize:
                num_of_norm_vals = len(normalize)
                assert num_of_norm_vals == 2 or num_of_norm_vals == 6
                mean = np.array(normalize[: num_of_norm_vals // 2])
                std = np.array(normalize[(num_of_norm_vals // 2) :])
            elif calc_normalization:
                # load train data and compute mean and std
                train_data_set = read_dataset(
                    datasets_dir, f"{dataset_name}_train", flatten=False
                )

                # average all axis except the color channel
                axis = tuple(np.arange(len(train_data_set.shape[:-1])))

                # calculate mean and std in double to avoid precision problems
                mean = np.mean(train_data_set.double(), axis).float()
                std = np.std(train_data_set.double(), axis).float()
            else:
                mean = None
                std = None

            print(f"\t\tmean: {mean}")
            print(f"\t\tstd: {std}")

            results = classifier_cls.grid_search(
                dataset_name, datasets_dir, output_dir, n_jobs, mean=mean, std=std
            )
            # get best result
            best_results[dataset_name] = sorted(
                results.as_dict()[dataset_name].items(), key=lambda e: e[1]
            ).pop()

        print(f"\n[+] Best Results")
        for dataset_name, (params, acc) in best_results.items():
            print(f"    -> {dataset_name}")
            print(f"       {params} @ {acc}")

    # train
    if command == "train":
        for dataset_name in datasets:
            classifier_cls.train_classifier(
                dataset_name, datasets_dir, output_dir, n_jobs, **classifier_args
            )

    # test
    if command == "test":
        assert len(datasets) == 1
        classifier_cls.test_classifier(
            classifier_name, datasets[0], datasets_dir, output_dir, n_jobs
        )


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        help="Command to execute. If choice is `train`, the hyperparameters need to be set accrodingly.",
        choices=["train", "test", "grid_search"],
        type=str,
    )
    parser.add_argument(
        "--n_jobs",
        help="Limits the number of cores used (if available).",
        type=int,
        default=cpu_count(),
    )
    parser.add_argument(
        "--datasets",
        help="Name of dataset(s).",
        action="append",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--datasets_dir",
        help="Directory containing the dataset(s).",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Working directory containing results and classifiers.",
        type=Path,
        required=True,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--normalize",
        nargs="+",
        type=float,
        metavar=("MEAN", "STD"),
        help="normalize with specified values for mean and standard deviation (either 2 or 6 values "
        "are accepted)",
    )
    group.add_argument(
        "--calc-normalization",
        action="store_true",
        help="calculates mean and standard deviation used in normalization from the training data",
    )

    parser.add_argument(
        "--classifier_name",
        help="Name of classifier (located within output directory). Only used when command set to "
        "'test'.",
        type=str,
    )

    subparsers = parser.add_subparsers(dest="baseline", help="Name of classifier.")
    subparsers.required = True

    knn_parser = subparsers.add_parser("knn", help="kNN-based classifier.")
    knn_parser.add_argument("--n_neighbors", type=int)

    prnu_parser = subparsers.add_parser("prnu", help="PRNU-based classifier.")
    prnu_parser.add_argument("--levels", type=int)
    prnu_parser.add_argument("--sigma", type=float)

    eigenfaces_parser = subparsers.add_parser(
        "eigenfaces", help="Eigenfaces-based classifier."
    )
    eigenfaces_parser.add_argument("--pca_target_variance", type=float)
    eigenfaces_parser.add_argument("--C", type=float)

    return parser.parse_args()


if __name__ == "__main__":
    import sklearnex

    sklearnex.patch_sklearn()
    main(**vars(parse_args()))
