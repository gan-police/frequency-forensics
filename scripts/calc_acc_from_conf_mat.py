import numpy as np

def calc_mean_std_accs(wavelet, mode):
    acc_lst = []
    known_acc_lst = []
    unknown_acc_lst = []

    for seed in range(4):
        matrix = np.load(open(f"confusion-matrix-lsun_bedroom_200k_png_log_packets_{wavelet}_boundary_missing_4_{mode}_{seed}-generalized.npy", "rb"))

        acc_lst.append((matrix[0, 0] + matrix[1:, 1].sum()) / matrix.sum())
        known_acc_lst.append((matrix[0, 0] + matrix[1:-1, 1].sum()) / matrix[:-1, :].sum())
        unknown_acc_lst.append(matrix[-1, 1] / matrix[-1, :].sum())

    max = (np.max(acc_lst), np.max(known_acc_lst), np.max(unknown_acc_lst))
    mean = (np.mean(acc_lst), np.mean(known_acc_lst), np.mean(unknown_acc_lst))
    std = (np.std(acc_lst), np.std(known_acc_lst), np.std(unknown_acc_lst))

    return max, mean, std

for mode in ["regression"]:
    print(f"{mode}:")
    for wavelet in ["db4"]:
        max, mean, std = calc_mean_std_accs(wavelet, mode)
        print(f"{wavelet} & {100*max[0]:.2f}\,\% & {100*mean[0]:.2f} $\pm$ {100*std[0]:.2f}\,\% & \
            {100*max[1]:.2f}\,\% & {100*mean[1]:.2f} $\pm$ {100*std[1]:.2f}\,\% & \
            {100*max[2]:.2f}\,\% & {100*mean[2]:.2f} $\pm$ {100*std[2]:.2f}\,\%")
