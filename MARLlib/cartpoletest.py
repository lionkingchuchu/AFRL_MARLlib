import ray
import gym
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.models import ModelCatalog

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# env config
max_neighborhood_size = 31
data_dir = '/workspace/precompute_distance_30/'
# Maximum number of steps in each episode
max_steps = 300  # need change
offset_start = True
offset_start_offset_min = 100
offset_start_offset_max = 8500
individual_rewards = True
# The number of times tune.report() has been called
training_iterations = 10  # changed
samples_per_iteration = 1
test_steps = 2
train_and_evaluate = True
run_or_experiment = "PPO"
# for local
gpu_count = 1
num_workers = 10  # need change
# for GOOGLE
# gpu_count = 0  # when running on CPU
# num_workers = 1
# specify the output directory here
out_dir = './custom_results/'

def rl_train(lr):
    ray.init()
    env_config = {
        # "max_steps": max_steps,
        # "max_neighborhood_size": max_neighborhood_size,
        # "individual_rewards": individual_rewards,
        # "offset_start": offset_start,
        # "offset_start_offset_min": offset_start_offset_min,
        # "offset_start_offset_max": offset_start_offset_max,
        # "data_directory": data_dir,
        # "log_steps": False,
        # "log_step_time": False
    }

    # if gpu_count > 0:
    #     num_gpus = 0.0001  # Driver GPU
    #     num_gpus_per_worker = (gpu_count - num_gpus) / num_workers
    # else:
    #     num_gpus_per_worker = 0

    config_PPO = {
        # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        "num_workers": 1,
        "num_envs_per_worker": num_workers,
        # "num_gpus_per_worker": 100,
        "rollout_fragment_length": 1,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 1,
        # "vtrace": False,
        "batch_mode": "complete_episodes",
        "shuffle_sequences": True,
        "train_batch_size": num_workers * max_steps,
        "preprocessor_pref": 'rllib',  # definitely faster than 'deepmind' from experiments
        "lr": tune.grid_search([0.0001, 5e-5, 5e-6]),
        "log_level": "WARN",
        "framework": "torch",
        "eager_tracing": True,
        "collect_metrics_timeout": 180,
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 10,
        "num_gpus": 1,  # need to change
        "num_cpus_per_worker": num_workers,
        "num_cpus_for_driver": 2,
        # "multiagent": {
        #     "policies": {
        #         # the first tuple value is None -> uses default policy
        #         # "aircrafts": (None, observation_space, action_space, {"gamma": 0.99}),
        #         # "basestations": (None, observation_space, action_space, {"gamma": 0.99}),
        #     },
        #     "policy_mapping_fn": select_policy
        # },
        "no_done_at_end": False,
        "env_config": env_config,
        "env": "CartPole-v0",
    }
  
    configs = {
        "PPO": config_PPO,
    }
    analysis = tune.run(
        run_or_experiment,
        num_samples=1,
        # scheduler=ASHAScheduler(metric="time_this_iter_s", mode="min"),
        stop={"training_iteration": training_iterations},
        reuse_actors=False,
        log_to_file=True,
        local_dir=out_dir,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        config=configs[run_or_experiment])

    

try:
    rl_train(5e-6)
except Exception as e:
    print(str(e))
finally:
    if ray.is_initialized():
        ray.shutdown()