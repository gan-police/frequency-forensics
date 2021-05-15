"""
As found at:
https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/baselines/knn.py
"""
from concurrent.futures import ProcessPoolExecutor
from .classifier import Classifier, read_dataset
from sklearn.neighbors import KNeighborsClassifier
from .utils import PersistentDefaultDict


class KNNClassifier(Classifier):

    def __init__(self, n_neighbors, n_jobs, **kwargs):
        super().__init__(**kwargs)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)

    def _fit(self, train_data, train_labels):
        self.knn.fit(train_data, train_labels)

    def _score(self, test_data, test_labels):
        return self.knn.score(test_data, test_labels)

    @staticmethod
    def grid_search(dataset_name, datasets_dir, output_dir, n_jobs):

        # hyperparameter grid
        knn_grid = [1] + [(2 ** x) + 1 for x in range(1, 11)]

        # init results
        results = PersistentDefaultDict(output_dir.joinpath(f'knn_grid_search.json'))

        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
        val_data, val_labels = read_dataset(datasets_dir, f'{dataset_name}_val')

        def grid_search_step(n_neighbors: int):
            knn_params_str = f'n_neighbors.{n_neighbors}'
            print(f"[+] {knn_params_str}")

            # skip if result already exists
            if dataset_name in results.as_dict() and \
                    knn_params_str in results.as_dict()[dataset_name]:
                return None

            # train and test classifier
            knn = KNNClassifier(n_neighbors, n_jobs)
            knn.fit(train_data, train_labels)
            return knn.score(val_data, val_labels)

        # start multiple processes to run the classifier in parallel for different parameters
        with ProcessPoolExecutor() as executor:
            for n_neighbors, score in zip(knn_grid, executor.map(grid_search_step, knn_grid)):
                if score is not None:
                    results[dataset_name, n_neighbors] = score

        return results

    @staticmethod
    def train_classifier(dataset_name, datasets_dir, output_dir, n_jobs, n_neighbors):
        results = PersistentDefaultDict(output_dir.joinpath(f'knn_test.json'))
        # classifier name
        classifier_name = f'classifier_{dataset_name}_knn_n_neighbors.{n_neighbors}'
        # load data
        train_data, train_labels = read_dataset(datasets_dir, f'{dataset_name}_train')
        test_data, test_labels = read_dataset(datasets_dir, f'{dataset_name}_test')
        # train classifier
        knn = KNNClassifier(n_neighbors, n_jobs)
        knn.fit(train_data, train_labels)
        # test classifier
        score = knn.score(test_data, test_labels)
        results[classifier_name] = score
