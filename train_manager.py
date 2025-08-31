import csv
from datetime import datetime
import random

import yaml

from train_diffusion_policy import train_by_steps
from rollout_manager import rollout
from train_dipper import train_dipper, rollout_dipper

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from run_scenarios import evaluate_all_scenarios

# import gdown


if __name__ == "__main__":
    # envs: ["antmaze-large-diverse-v1", "pointmaze-medium-v2"]

    # train(
    #     env_id="antmaze-large-diverse-v1",
    #     rollouts=False
    # )
    # evaluate('antmaze_90.pt', 'antmaze-large-diverse-v1')
    # evaluate('antmaze_200.pt', 'antmaze-large-diverse-v1')

    # Point Maze - no map no goal
    # train(
    #     checkpoint='pointmaze_90.pt',
    #     env_id="pointmaze-medium-v2",
    #     debug=False,
    #     rollouts=False,
    #     goal_conditioned=False,
    #     num_epochs=200
    # )
    # evaluate('pointmaze_200.pt', 'pointmaze-medium-v2')
    # env_id = "pointmaze-medium-v2"
    # evaluate_all_scenarios(
    #     env_id='pointmaze-large-v2',
    #     scenarios_idx=[0,1,2],
    #     runs_per_scenario=1,
    #     debug=True,
    #     checkpoint='pointmaze_goal_local_map_grid_encoder_15_09_19_48_20.pt')

    # evaluate_all_scenarios(
    #     env_id='pointmaze-medium-v2',
    #     scenarios_idx=[2],
    #     runs_per_scenario=1,
    #     debug=True,
    #     checkpoint='pointmaze_goal_local_map_grid_encoder_15_09_19_48_20.pt')
    # Point Maze - map and goal conditioned

    # debug = False
    # rollouts = True
    # position_conditioned = False
    # goal_conditioned = True
    # local_map_conditioned = True
    # local_map_scale = 0.2
    # local_map_embedding_dim = 64
    # map_encoder = "grid"  # "grid", "max", "identity", "mlp"
    # unet_size = 'medium'  # 'small', 'medium', 'large'
    # num_epochs = 30  #100

    # augmentations = []  # ["rotate", "mirror"]
    env_id = "carmaze"  # "pointmaze-large-v2" "dronemaze" "droneforest" "carmaze" "antmaze"
    unet_dims = {
        'small': [64, 128, 256],
        'medium': [256, 512, 1024],
        'large': [512, 1024, 2048],
        'xlarge': [1024, 2048, 4096]
    }

    with open(f"cfgs/{env_id}.yaml", "r") as file:
        loaded_config = yaml.safe_load(file)

    timestamp = datetime.now().strftime('%d_%m_%H_%M')
    experiment_name = f"carmaze_medium_{timestamp}"  # _{timestamp}

    train_by_steps(
        # checkpoint="antmaze_action_prediction_map_02x2002_03_22_44_epoch_15_step_25000.pt",
        experiment_name=experiment_name,
        debug=loaded_config.get('debug', False),
        prediction_type=loaded_config.get('prediction_type', "actions"),    #  ["actions","observations"]
        obs_history=loaded_config.get('obs_history', 1),
        action_history=loaded_config.get('action_history', 1),
        goal_conditioned=loaded_config.get('goal_conditioned', True),
        local_map_conditioned=loaded_config.get('local_map_conditioned', True),
        local_map_size=loaded_config.get('local_map_size', 20),
        local_map_scale=loaded_config.get('local_map_scale', 0.2),
        local_map_embedding_dim=loaded_config.get('local_map_embedding_dim', 400),
        local_map_encoder=loaded_config.get('local_map_encoder', "resnet"), #["grid", "max", "identity", "mlp","resnet"]
        num_epochs=loaded_config.get('num_epochs', 15),
        checkpoint_every=1000,
        rollout_every=1000,
        env_id=loaded_config.get('env_id', "carmaze"),
        augmentations=loaded_config.get('augmentations', ["mirror"]),   # ["rotate", "mirror"]
        policy=loaded_config.get('policy', "flow_matching"),    # ["diffusion","flow_matching"]
        num_diffusion_iters=loaded_config.get('num_diffusion_iters', 5),
        unet_down_dims=unet_dims[loaded_config.get('unet_down_dims', 'large')],
        pred_horizon=loaded_config.get('pred_horizon', 64),
        action_horizon=loaded_config.get('action_horizon', 8),
    )

    # evaluate(checkpoint="carmaze_psi_normalized_03_02_13_46_epoch_5_step_4000.pt",
    #          env_id=env_id,
    #          policy="flow_matching",
    #          num_episodes=1,
    #          render_mode='human',
    #          num_diffusion_iters=10,
    #          output_dir='checkpoints/',
    #          action_horizon=2,
    #          obs_horizon=1,
    #          action_history=1,
    #          position_conditioned=False,
    #          goal_conditioned=True,
    #          goal_dim=2,
    #          local_map_conditioned=True,
    #          local_map_size=local_map_size,
    #          local_map_scale=local_map_scale,
    #          local_map_embedding_dim=256,
    #          local_map_encoder="resnet",
    #          unet_down_dims=unet_dims['large'],
    #          )

    # config = OmegaConf.create({
    #     'env_id': 'PointMaze_Large-v3',
    #     'diffusion_model': 'diffusion_forcing',
    #     'num_episodes': 3,
    #     'render_mode': 'human',
    #     'num_diffusion_iters': 100,
    #     'output_dir': 'checkpoints/',
    #     'action_horizon': 8,
    #     'position_conditioned': False,
    #     'goal_conditioned': True,
    #     'goal_dim': 2,
    #     'local_map_conditioned': True,
    #     'local_map_size': 10,
    #     'local_map_scale': 0.2,
    #     'local_map_embedding_dim': 400,
    #     'local_map_encoder': map_encoder,  # Assign the variable directly
    # })
    #
    # config_dipper = OmegaConf.create({
    #     'env_id': 'PointMaze_Large-v3',
    #     'diffusion_model': 'diffusion_forcing',
    #     'num_episodes': 3,
    #     'render_mode': 'human',
    #     'num_diffusion_iters': 100,
    #     'output_dir': 'checkpoints/',
    #     'action_horizon': 8,
    #     'position_conditioned': False,
    #     'goal_conditioned': True,
    #     'goal_dim': 2,
    #     'local_map_conditioned': True,
    #     'local_map_size': 10,
    #     'local_map_scale': 0.2,
    #     'local_map_embedding_dim': 252,
    #     'local_map_encoder': map_encoder,  # Assign the variable directly
    # })

    # evaluate('pointmaze_goal_local_map_grid_encoder_15_09_19_48_20.pt', 'pointmaze-large-v2',
    #          num_episodes=1, render_mode='human',
    #          local_map_encoder=local_map_encoder,
    #          local_map_embedding_dim=local_map_embedding_dim)

    # timestamp = datetime.now().strftime('%d_%m_%H_%M')
    # experiment_name = f"pointmaze_small_resnet_{timestamp}"  # _{timestamp}
    #
    # train(
    #     experiment_name=experiment_name,
    #     datasets="pointmaze-large-v2",
    #     debug=debug,
    #     rollouts=True,
    #     position_conditioned=False,
    #     goal_conditioned=True,
    #     local_map_conditioned=True,
    #     local_map_size=local_map_size,  # 10
    #     local_map_scale=local_map_scale,  # 0.2
    #     local_map_embedding_dim=256,    #64,  # 400,  # 64,
    #     local_map_encoder="resnet",  # "grid", "max", "identity", "mlp", "resnet"
    #     num_epochs=num_epochs,
    #     checkpoint_every=10,
    #     rollout_every=1,
    #     env_id="pointmaze-large-v2",
    #     augmentations=["rotate", "mirror"],  # ["rotate", "mirror"]
    #     unet_down_dims=[64, 128, 256],  # [256, 512, 1024]
    #     pred_horizon=64,
    # )

    # timestamp = datetime.now().strftime('%d_%m_%H_%M')
    # experiment_name = f"pointmaze_small_cnn_{timestamp}"  # _{timestamp}
    # train(
    #     checkpoint="pointmaze_small_cnn_07_11_10_59_epoch_40.pt",
    #     experiment_name=experiment_name,
    #     datasets="pointmaze-large-v2",
    #     debug=debug,
    #     rollouts=True,
    #     position_conditioned=False,
    #     goal_conditioned=True,
    #     local_map_conditioned=True,
    #     local_map_size=local_map_size,  # 10
    #     local_map_scale=local_map_scale,  # 0.2
    #     local_map_embedding_dim=256,    #64,  # 400,  # 64,
    #     local_map_encoder="grid",  # "grid", "max", "identity", "mlp"
    #     num_epochs=num_epochs,
    #     checkpoint_every=10,
    #     rollout_every=1,
    #     env_id="pointmaze-large-v2",
    #     augmentations=["rotate", "mirror"],  # ["rotate", "mirror"]
    #     unet_down_dims=[64, 128, 256],  # [256, 512, 1024]
    #     pred_horizon=64,
    # )
    # timestamp = datetime.now().strftime('%d_%m_%H_%M')
    # experiment_name = f"pointmaze_medium_resnet_{timestamp}"  # _{timestamp}
    #
    # train(
    #     checkpoint="pointmaze_medium_resnet_08_11_07_59_epoch_10.pt",
    #     experiment_name=experiment_name,
    #     datasets="pointmaze-large-v2",
    #     debug=debug,
    #     rollouts=True,
    #     position_conditioned=False,
    #     goal_conditioned=True,
    #     local_map_conditioned=True,
    #     local_map_size=local_map_size,  # 10
    #     local_map_scale=local_map_scale,  # 0.2
    #     local_map_embedding_dim=256,    #64,  # 400,  # 64,
    #     local_map_encoder="resnet",  # "grid", "max", "identity", "mlp", "resnet"
    #     num_epochs=num_epochs,
    #     checkpoint_every=10,
    #     rollout_every=1,
    #     env_id="pointmaze-large-v2",
    #     augmentations=["rotate", "mirror"],  # ["rotate", "mirror"]
    #     unet_down_dims=[256, 512, 1024],  # [256, 512, 1024]
    #     pred_horizon=64,
    # )
    #
    # timestamp = datetime.now().strftime('%d_%m_%H_%M')
    # experiment_name = f"pointmaze_medium_cnn_{timestamp}"  # _{timestamp}
    # train(
    #     experiment_name=experiment_name,
    #     datasets="pointmaze-large-v2",
    #     debug=debug,
    #     rollouts=True,
    #     position_conditioned=False,
    #     goal_conditioned=True,
    #     local_map_conditioned=True,
    #     local_map_size=local_map_size,  # 10
    #     local_map_scale=local_map_scale,  # 0.2
    #     local_map_embedding_dim=256,    #64,  # 400,  # 64,
    #     local_map_encoder="grid",  # "grid", "max", "identity", "mlp"
    #     num_epochs=num_epochs,
    #     checkpoint_every=10,
    #     rollout_every=1,
    #     env_id="pointmaze-large-v2",
    #     augmentations=["rotate", "mirror"],  # ["rotate", "mirror"]
    #     unet_down_dims=[256, 512, 1024],  # [256, 512, 1024]
    #     pred_horizon=64,
    # )
    # train_dipper(
    #     # checkpoint='pointmaze_large_goal_local_map_grid_encoder_26_10_14_15_epoch_40.pt',
    #     experiment_name=experiment_name,
    #     datasets="pointmaze-large-v2",
    #     debug=False,
    #     rollouts=True,
    #     map_encoder=map_encoder,  # "grid", "max", "identity", "mlp"
    #     map_embedding_dim=144,
    #     num_epochs=200,
    #     checkpoint_every=5,
    #     rollout_every=5,
    #     env_id="pointmaze-large-v2",
    #     augmentations=[],  # ["rotate", "mirror"]
    #     pred_horizon=200,
    # )
    # evaluate('pointmaze_large_goal_local_map_mlp_encoder_18_09_11_48_epoch_2.pt',
    #          env_id,
    #          num_episodes=3,
    #          render_mode='human',
    #          num_diffusion_iters=100,
    #          output_dir='checkpoints/',
    #          action_horizon=8,
    #          position_conditioned=False,
    #          goal_conditioned=True,
    #          goal_dim=2,
    #          local_map_conditioned=True,
    #          local_map_size=10,
    #          local_map_scale=0.2,
    #          local_map_embedding_dim=400,
    #          local_map_encoder=local_map_encoder,
    #          )
    # evaluate('pointmaze_200.pt', 'pointmaze-medium-v2')
    # evaluate_all_scenarios(
    #     env_id='pointmaze-medium-v2',
    #     scenarios_idx=[0],
    #     runs_per_scenario=1,
    #     debug=True,
    #     checkpoint='pointmaze_large_goal_local_map_mlp_encoder_18_09_13_21_epoch_18.pt')
