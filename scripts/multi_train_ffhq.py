# Running this script in sbatch will train multiple neural networks on the same gpu.
import time
import datetime

import subprocess
subprocess.call('pwd')

print('running jobs in parallel')

# experiment_lst = \
#     [["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
#      "0", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_log_packets_haar_boundary", "--nclasses", "3"],
#      ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
#      "1", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_log_packets_haar_boundary", "--nclasses", "3"],
#      ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
#      "2", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_log_packets_haar_boundary", "--nclasses", "3"],
#      ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
#      "3", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_log_packets_haar_boundary", "--nclasses", "3"],
#      ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
#      "4", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_log_packets_haar_boundary", "--nclasses", "3"]]
# jobs = []
# for exp_no, experiment in enumerate(experiment_lst):
#     time.sleep(10)
#     time_str = str(datetime.datetime.today())
#     print(experiment, ' at time:', time_str)
#     with open("./log/out/" + time_str + ".txt", "w") as f:
#         jobs.append(subprocess.Popen(experiment, stdout=f))
# for job in jobs:
#     job.wait()


experiment_lst = \
    [["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
      "0", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_raw", "--nclasses", "3"],
     ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
       "1", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_raw", "--nclasses", "3"],
     ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
       "2", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_raw", "--nclasses", "3"],
     ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
       "3", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_raw", "--nclasses", "3"],
     ["python", "-m", "freqdect.train_classifier", "--features", "packets", "--seed",
       "4", "--data-prefix", "/nvme/mwolter/ffhq128/source_data_raw", "--nclasses", "3"]]
jobs = []
for exp_no, experiment in enumerate(experiment_lst):
    time.sleep(10)
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    with open("./log/out/" + time_str + ".txt", "w") as f:
        jobs.append(subprocess.Popen(experiment, stdout=f))
for job in jobs:
    job.wait()

print('done')