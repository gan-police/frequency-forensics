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


def main():
    raw_logs = pickle.load(open('./log/packetsFalse.pkl', 'rb'))
    packet_logs = pickle.load(open('./log/packetsTrue.pkl', 'rb'))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # raw_train_loss_steps, raw_train_loss = zip(*raw_logs[-1]['train loss'])
    # steps, mean, std = stack_list(raw_logs, key='train loss')
    # raw_train_acc_steps, raw_train_acc = zip(*raw_logs[-1]['train acc'])
    steps, mean, std = get_plot_tuple(raw_logs, 'train acc')
    # plot_mean_std(steps, mean, std, color=colors[0],
    #               label='raw train acc')

    # raw_val_acc_steps, raw_val_acc = zip(*raw_logs[-1]['val acc'])
    steps, mean, std = get_plot_tuple(raw_logs, 'val acc')
    plot_mean_std(steps, mean, std, color=colors[0],
                  label='raw validation acc')

    # packet_train_acc_steps, packet_train_acc = zip(*packet_logs[-1]['train acc'])
    # steps, mean, std = get_plot_tuple(packet_logs, 'train acc')
    # plot_mean_std(steps, mean, std, color=colors[2],
    #               label='packet val acc')
    # packet_val_acc_steps, packet_val_acc = zip(*packet_logs[-1]['val acc'])
    steps, mean, std = get_plot_tuple(packet_logs, 'val acc')
    plot_mean_std(steps, mean, std, color=colors[1],
                  label='packet validation acc')

    plt.ylabel('accuracy')
    plt.xlabel('training steps')
    plt.title('Validation accuracy FFHQ')
    plt.legend()
    plt.show()

    print('stop')


if __name__ == '__main__':
    main()
