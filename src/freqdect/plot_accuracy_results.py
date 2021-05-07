import pickle
import numpy as np
import matplotlib.pyplot as plt


def stack_list(dict_list: {}, key: str):
    step_lst = []
    acc_lst = []
    for current_dictionary in dict_list:
        steps, acc = zip(*current_dictionary[key])
        step_lst.append(steps)
        acc_lst.append(acc)
    return np.stack(step_lst), np.stack(acc_lst)


def get_steps_mean_std(step_lst, cost_lst):
    mean = np.mean(cost_lst, axis=0)
    std = np.std(cost_lst, axis=0)
    return step_lst[0, :], mean, std


def get_plot_tuple(dict_list: {}, key: str):
    steps, loss = stack_list(dict_list, key)
    steps, mean, std = get_steps_mean_std(steps, loss)
    return steps, mean, std


def plot_mean_std(steps, mean, std, color, label='', marker='.'):
    plt.plot(steps, mean, label=label, color=color, marker=marker)
    plt.fill_between(steps, mean - std,
                     mean + std,
                     color=color, alpha=0.2)


def get_test_acc_mean_std(dict_list: dict, key: str):
    test_accs = []
    for experiment_dict in dict_list:
        test_accs.append(experiment_dict[key])
    return np.mean(test_accs), np.std(test_accs)


def main():
    raw_logs = pickle.load(open('./log/packetsFalse.pkl', 'rb'))
    packet_logs = pickle.load(open('./log/packetsTrue.pkl', 'rb'))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    steps, mean, std = get_plot_tuple(raw_logs, 'train_acc')

    steps, mean, std = get_plot_tuple(raw_logs, 'val_acc')
    plot_mean_std(steps, mean, std, color=colors[0],
                  label='raw validation acc')

    steps, mean, std = get_plot_tuple(packet_logs, 'val_acc')
    plot_mean_std(steps, mean, std, color=colors[1],
                  label='packet validation acc')

    pt_mean, pt_std = get_test_acc_mean_std(packet_logs, 'test_acc')
    rt_mean, rt_std = get_test_acc_mean_std(raw_logs, 'test_acc')
    plt.errorbar(steps[-1], pt_mean, pt_std, color=colors[2],
                 label='packet test acc', marker='.')
    plt.errorbar(steps[-1], rt_mean, rt_std, color=colors[3],
                 label='raw test acc', marker='.')

    plt.ylabel('mean accuracy')
    plt.xlabel('training steps')
    plt.title('Validation accuracy FFHQ-StyleGAN')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
