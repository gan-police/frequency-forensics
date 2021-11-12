# Running this script in sbatch will train multiple neural networks on the same gpu.
import time
import datetime
import subprocess
subprocess.call('pwd')

print('running jobs in parallel')

experiment_lst = []
for model in ['cnn']:
    for wavelet_str in ["sym3", "sym4", "sym5", "db3", "db4", "db5"]:
        for seed in [0, 1, 2, 3, 4]:
            experiment_lst.append(
                (["python", "-m", "freqdect.train_classifier", "--features", "packets",
                 "--model", str(model),
                 "--seed", str(seed),
                 "--data-prefix",
                 "/nvme/mwolter/celeba/celeba_align_png_cropped_log_packets_" + wavelet_str + "_boundary",
                 "--nclasses", "5"], (str(model), str(seed), wavelet_str)))
jobs = []    
for exp_no, experiment in enumerate(experiment_lst):
    time.sleep(10)
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    file_name = f"{time_str}_celebA_{experiment[1][0]}_{experiment[1][1]}_{experiment[1][2]}.txt"
    with open(f"./log/out/{file_name}", "w") as file:
        jobs.append(subprocess.Popen(experiment[0], stdout=file))
    if exp_no % 5 == 0 and exp_no > 0:
        for job in jobs:
            job.wait()
        jobs = []
