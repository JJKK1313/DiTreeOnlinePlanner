import csv
import os
import time
from datetime import datetime
import random

from playsound import playsound
import yaml
import sys
from scipy.ndimage import distance_transform_edt
import matplotlib

USE_AGG = True
if USE_AGG:  # prevent crashing because of main loop thread with IDE thread.
    matplotlib.use('Agg')
# Detect if in debug model
gettrace = getattr(sys, 'gettrace', None)
# debug = False
# if gettrace is not None:
#     debug = gettrace() is not None

debug = False
SHOW_PLOTS = False
BLOCK_PLOTS = False
print(f'\033[94m Debug Mode Enabled: {debug} \033[00m')

import matplotlib.pyplot as plt
import numpy as np

import minari
import pandas as pd
import torch

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from drone_env import DroneEnv
from car_env import CarEnv

from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_diffusion_policy import init_noise_pred_net
# from train_dipper import init_dipper_net  # NOT IN USE

from policies.fm_policy import DiffusionSampler
from policies.uniform_policy import UniformSampler

from planners.RRT import RRT_Planner
from planners.random_tree import RandomTreePlanner
from planners.MPC import MPC_Planner

import common.map_utils
from obstacle_insertion import insert_obstacle_to_maze
# from plot_manager import plot_manager, plot_logger
from plot_logger import plot_logger


def plan_path(planner, curr_state, goal_state, stats2keep: dict, time_budget=None):
    planner.plan_count += 1
    prev_time_budget = planner.time_budget
    planner.reset(start_state=curr_state, goal_state=goal_state)
    if time_budget is not None:
        planner.time_budget = time_budget
    main_path_array, main_actions = planner.plan()
    for k in stats2keep.keys():
        stats2keep[k] += planner.results[k]
    planner.reset(start_state=curr_state, goal_state=goal_state)
    planner.time_budget = prev_time_budget
    return main_path_array, main_actions


# Function to run the experiment
def run_experiment(planner, num_runs=100):
    success_count = 0
    runtimes = []
    iterations = []
    path_lengths = []
    path_avg_speeds = []
    number_of_nodes = []

    for _ in range(num_runs):
        planner.reset()
        path, actions = planner.plan()
        curr_iterations = planner.results["iterations"]
        runtime = planner.results["time"]
        num_nodes = planner.results["number_of_nodes"]

        if path is not None:
            path_length = 0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(path[i + 1, :2] - path[i, :2])
            avg_speed = np.mean(np.linalg.norm(path[:, 3:5], axis=1))

            success_count += 1
        runtimes.append(runtime)
        iterations.append(curr_iterations)
        path_lengths.append(path_length)
        path_avg_speeds.append(avg_speed)
        number_of_nodes.append(num_nodes)

    success_rate = success_count / num_runs
    return success_rate, runtimes, iterations, path_lengths, path_avg_speeds, number_of_nodes


def scan_and_update_maze(planner, maze_data, maze_data_with_obstacle, scanned_maze, debug=False):
    curr_state = planner.env.state
    curr_state_xy = curr_state.copy()
    curr_state_xy[:2] = planner.env.cell_xy_to_rowcol(curr_state[:2], floor_enable=False)
    curr_state_xy[:2] = curr_state_xy[:2][::-1]
    distances, endpoints, visited_points = planner.env.lidar2dsim.scan(curr_state_xy[:3], maze_data_with_obstacle,
                                                                       debug)
    endpoints = np.floor(endpoints).astype('int')  # remove samples which didn't encounter obstacles
    maze_data[endpoints[:, 1], endpoints[:, 0]] = 1
    scanned_maze[visited_points[:, 1], visited_points[:, 0]] = 2  # set points to be visited
    scanned_maze[endpoints[:, 1], endpoints[:, 0]] = 1  # set points to be obstacles
    planner.update_maze(maze_data)
    if debug:
        plt.figure()
        plt.imshow(scanned_maze, origin='lower', extent=[0, scanned_maze.shape[0], 0, scanned_maze.shape[1]])
        plt.show()


def plot_traj(planner, maze_data, states_of_action, block=False, show_plots=False, title='', path=None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(maze_data, origin='lower', extent=[0, maze_data.shape[0], 0, maze_data.shape[1]])
    colors = plt.cm.cool(np.linspace(0, 1, len(states_of_action)))
    for i, s in enumerate(states_of_action):
        s = s.copy()
        s[:2] = planner.env.cell_xy_to_rowcol(s[:2], floor_enable=False)
        s[:2] = s[:2][::-1]
        plt.scatter(s[0], s[1], color=colors[i], s=5)
        plt.quiver(s[0], s[1], np.cos(-s[2]), np.sin(-s[2]), scale=20)
    plt.title(title)
    plt.subplot(1, 2, 2)
    plt.imshow(planner.env.prob_map, origin='lower', extent=[0, maze_data.shape[0], 0, maze_data.shape[1]])
    if path is not None and isinstance(path, str):
        plt.savefig(path, dpi=200)
        # plot_logger.save_fig(plt.gcf(), path)
        plt.close(plt.gcf())
        # plot_manager.save(path, dpi=200)
    if show_plots:
        if not block:
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        else:
            plt.show()


def check_no_obstacles_in_path(planner, scanned_maze, main_path_array, debug=False):
    main_path_points_rowcol = np.array(list(
        map(lambda x: planner.env.cell_xy_to_rowcol(x, floor_enable=False), main_path_array[:, :2]))
    )
    main_path_points_rc_round = np.floor(main_path_points_rowcol[:, ::-1]).astype('int')
    obstacle_in_path_loc_idx = -1
    for idx, p in enumerate(main_path_points_rc_round):
        if scanned_maze[p[1], p[0]] == 1:
            obstacle_in_path_loc_idx = idx
            break
    if debug:
        plt.figure()
        plt.imshow(scanned_maze, origin='lower', extent=[0, scanned_maze.shape[0], 0, scanned_maze.shape[1]])
        colors = plt.cm.cool(np.linspace(0, 1, len(main_path_points_rowcol)))
        for i, p in enumerate(main_path_points_rowcol):
            plt.scatter(p[1], p[0], color=colors[i], s=5)
        if obstacle_in_path_loc_idx >= 0:
            plt.scatter(main_path_points_rowcol[obstacle_in_path_loc_idx, 1],
                        main_path_points_rowcol[obstacle_in_path_loc_idx, 0],
                        color='k',
                        s=5)
        plt.title("Path in Obstacle in Black Dot (if exist)")
        plt.show()
    return obstacle_in_path_loc_idx


def evaluate_all_scenarios(mazes_dir, scenarios_file, cfg_file, obstacle_dims=[2, 1],
                           total_runs=100, offline_time_budget=60,
                           online_time_budget=2,
                           diffusion_sampler_checkpoints=None,
                           root_folder='benchmark_results/ditree',
                           obstacle_loc_row_col=None):
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    unet_dims = {
        'small': [64, 128, 256],
        'medium': [256, 512, 1024],
        'large': [512, 1024, 2048],
        'xlarge': [1024, 2048, 4096]
    }

    with open(f"cfgs/{cfg_file}.yaml", "r") as file:
        loaded_config = yaml.safe_load(file)

    # debug = loaded_config.get('debug', False)
    prediction_type = loaded_config.get('prediction_type', "actions")  # ["actions","observations"]
    obs_history = loaded_config.get('obs_history', 1)  # get only current and previous actions and states.
    action_history = loaded_config.get('action_history', 1)
    position_conditioned = False
    goal_conditioned = loaded_config.get('goal_conditioned', True)
    local_map_conditioned = loaded_config.get('local_map_conditioned', True)
    local_map_size = loaded_config.get('local_map_size', 20)
    local_map_scale = loaded_config.get('local_map_scale', 0.2)
    local_map_embedding_dim = loaded_config.get('local_map_embedding_dim', 400)
    env_id = loaded_config.get('env_id', "carmaze")
    policy = loaded_config.get('policy', "flow_matching")  # ["diffusion","flow_matching"]
    num_diffusion_iters = loaded_config.get('planning_diffusion_iters', 5)
    unet_down_dims = unet_dims[loaded_config.get('denoiser_size', 'large')]
    pred_horizon = loaded_config.get('pred_horizon', 64)
    action_horizon = loaded_config.get('action_horizon', 8)
    goal_conditioning_bias = loaded_config.get('goal_conditioning_bias', 0.85)
    prop_duration = loaded_config.get('prop_duration', [64])

    goal_dim = 2
    s_global = 1.0
    full_obs_dim = 6
    obs_dim = full_obs_dim
    if not position_conditioned:
        obs_dim -= 3  # remove (x,y,theta)
    action_dim = 2

    # render_mode = 'human' if debug else 'rgb_array'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'checkpoints/'

    noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True, prediction_type='epsilon')

    noise_pred_net = init_noise_pred_net(
        input_dim=action_dim if prediction_type == "actions" else full_obs_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder="resnet",
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        down_dims=unet_down_dims,
    )

    checkpoint = torch.load(output_dir + diffusion_sampler_checkpoints['resnet'])
    noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])
    noise_pred_net = noise_pred_net.to(device).eval()

    diffusion_sampler_small_resnet = DiffusionSampler(noise_pred_net, noise_scheduler, env_id,
                                                      policy='flow_matching',
                                                      pred_horizon=pred_horizon, action_dim=action_dim,
                                                      prediction_type=prediction_type,
                                                      obs_history=obs_history, action_history=action_history,
                                                      goal_conditioned=True, num_diffusion_iters=num_diffusion_iters,
                                                      local_map_size=local_map_size).eval()

    os.makedirs(root_folder, exist_ok=True)
    dirs_names = os.listdir(root_folder)
    # the index of the current run taken from directories names.
    iterations = list(map(lambda x: int(x), dirs_names))
    current_run = max(iterations) + 1 if len(iterations) > 0 else 1

    # Load the scenarios from the CSV file
    scenarios_df = pd.read_csv(scenarios_file)
    # Iterate through all scenarios
    for scenario_idx in range(len(scenarios_df)):
        scenario_name = scenarios_df.loc[scenario_idx]['scenario_name']
        maze_name = scenarios_df.loc[scenario_idx]['maze_name']
        start_row = scenarios_df.loc[scenario_idx]['start_row']
        start_col = scenarios_df.loc[scenario_idx]['start_col']
        start_deg = scenarios_df.loc[scenario_idx]['start_deg']
        goal_row = scenarios_df.loc[scenario_idx]['goal_row']
        goal_col = scenarios_df.loc[scenario_idx]['goal_col']

        scenario_folder = f'{root_folder}/{current_run}/{scenario_name}/'
        os.makedirs(scenario_folder, exist_ok=True)

        # Load the corresponding maze
        maze_path = os.path.join(mazes_dir, f'{maze_name}.csv')
        if not os.path.exists(maze_path):
            print(f"Maze file {maze_path} not found, skipping scenario.")
            continue

        # Load maze without obstacles.
        maze_data_original = np.loadtxt(maze_path, delimiter=',')
        maze_data = maze_data_original.copy()

        # Create a new environment for each maze with the maze data
        env = CarEnv(maze_map=maze_data, collision_checking=False)  # Collision check is done in RRT Planners

        # (for convenience).
        start_xy = env.cell_rowcol_to_xy(np.array([int(start_row), int(start_col)]))
        start_rad = np.deg2rad(float(start_deg))
        goal_xy = env.cell_rowcol_to_xy(np.array([int(goal_row), int(goal_col)]))
        start = np.array([start_xy[0], start_xy[1], start_rad, 0.0, 0.0, 0.0])
        goal = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0, 0.0, 0.0])

        # Load maze insert obstacle by hand
        if obstacle_loc_row_col is None:
            maze_data_with_obstacle = insert_obstacle_to_maze(maze_data_original, obstacle_dims=obstacle_dims,
                                                              start=np.array([int(start_row), int(start_col)]),
                                                              goal=np.array([int(goal_row), int(goal_col)]))
        else:
            maze_data_with_obstacle = maze_data_original.copy()
            y_min = max(obstacle_loc_row_col[0], 0)
            y_max = min(obstacle_loc_row_col[0] + obstacle_dims[0], maze_data.shape[0])
            x_min = max(obstacle_loc_row_col[1], 0)
            x_max = min(obstacle_loc_row_col[1] + obstacle_dims[1], maze_data.shape[1])
            maze_data_with_obstacle[y_min:y_max, x_min:x_max] = 1

        plt.figure()
        plt.imshow(2 * maze_data_with_obstacle - maze_data_original, origin='lower',
                   extent=[0, maze_data_original.shape[0], 0, maze_data_original.shape[1]])
        plt.savefig(f'{scenario_folder}/Inserted_Obstacle', dpi=200)

        # Create planner
        diffusion_RRT = RRT_Planner(start, goal, env_id=env_id, environment=env,
                                    sampler=diffusion_sampler_small_resnet,
                                    prediction_type=prediction_type,
                                    action_horizon=action_horizon,
                                    # edge_length=edge_length,
                                    local_map_size=local_map_size,
                                    local_map_scale=local_map_scale,
                                    global_map_scale=s_global,
                                    goal_conditioning_bias=goal_conditioning_bias,
                                    prop_duration=prop_duration,
                                    offline_time_budget=offline_time_budget,
                                    time_budget=online_time_budget,
                                    max_iter=300,
                                    verbose=True,
                                    debug=debug,
                                    scenario_num=current_run, scenario_name=scenario_name,
                                    root_folder=scenario_folder
                                    )

        planner_name = "diffusion_RRT_PD64"
        planner = diffusion_RRT
        print(f"Running scenario\033[92m {scenario_name}\033[00m with planner\033[92m {planner_name}\033[00m...")
        # Prepare CSV output for each scenario and planner
        scenario_output_csv = f'{scenario_folder}/{planner_name}_{env_id}.csv'

        existing_rows = 0
        # Get number of scenarios that were done already
        if os.path.exists(scenario_output_csv):
            with open(scenario_output_csv, mode='r', newline='') as file:
                reader = csv.reader(file)
                rows = list(reader)
                if len(rows) > 1:  # Exclude the header
                    existing_rows = len(rows) - 1  # Exclude the header

        remaining_runs = total_runs - existing_rows
        if remaining_runs <= 0:
            print(f"CSV already contains {existing_rows} rows. No additional runs needed.")
            return

        print(f"CSV contains {existing_rows} rows. Running {remaining_runs} more iterations.")
        # If the file is empty, create it and add headers
        if existing_rows == 0:
            with open(scenario_output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['iteration', 'success', 'runtime [sec]', 'trajectory_length', 'trajectory_time',
                     'avg_velocity', 'num_states_in_tree', 'num_RRT_iterations',
                     'ctrl_effort_max', 'ctrl_effort_mean', 'ctrl_effort_std'])
        # Run only the remaining required iterations
        total_cc = 0
        ALLOWED_TRIALS = 5
        for i in range(existing_rows, existing_rows + remaining_runs):
            print(f"\033[91m Scenario: {scenario_name}, Iteration: {i + 1}/{total_runs}\033[00m")
            print(f"Init...")
            planner.scenario_iter_num = str(i + 1)
            executed_path = []
            executed_actions = []
            stats_2_keep = dict(number_of_nodes=0,
                                iterations=0)
            planner.plan_count = -1  # main path is going to be plan 0
            done = False
            curr_state = np.array([start_xy[0], start_xy[1], start_rad, 0, 0, 0])
            common.map_utils.cc_calls = 0
            maze_data = maze_data_original.copy()
            scanned_maze = maze_data_original.copy()
            start_time = time.time()

            # Initial Scan
            print("Initial Scan...")
            planner.reset(start_state=curr_state, goal_state=goal, reset_main_path=True)
            scan_and_update_maze(planner, maze_data, maze_data_with_obstacle, scanned_maze)

            # Main Path to do
            print("Main Path Planning...")
            main_path_array, main_actions = plan_path(planner, curr_state, goal, stats2keep=stats_2_keep,
                                                      time_budget=offline_time_budget)
            if main_path_array is None:
                continue
            planner.init_main_path = main_path_array.copy()

            if main_actions is None:
                print('\033[91m', 10 * '*', "Got into a blocked path and cant find a way out!!!", 10 * '*',
                      '\033[00m')
            else:
                # plot_traj(planner, scanned_maze, planner.init_main_path, block=BLOCK_PLOTS, show_plots=SHOW_PLOTS,
                #           title="Reference Path",
                #           path=f'{root_folder}/{current_run}/{scenario_name}/Iter_{i + 1}/Ref_Plan')
                planner.env.reset_done()
                action_idx = 0
                trials = 0
                obstacles_in_way = -1
                prev_plan_time = time.time()
                while not planner.env.is_done(curr_state):
                    if (obstacles_in_way >= 0 or action_idx == main_actions.shape[0]
                            or (time.time() - prev_plan_time) >= 2):
                        if obstacles_in_way >= 0:
                            print("Obstacles in way found!", end=' ')
                        print("Need new plan...")
                        main_actions = None  # reset actions
                        while trials < ALLOWED_TRIALS and main_actions is None:
                            main_path_array, main_actions = plan_path(planner, curr_state, goal,
                                                                      stats2keep=stats_2_keep)
                            if main_actions is None:
                                print('\033[91m', 10 * '*', "Got into a blocked path and cant find a way out!!!",
                                      10 * '*', '\033[00m')
                                trials += 1
                        if trials >= ALLOWED_TRIALS:
                            break
                        trials = 0

                        planner.env.reset_done()
                        planner.reset(start_state=curr_state, goal_state=goal)
                        action_idx = 0
                        # plot_traj(planner, scanned_maze, main_path_array, block=BLOCK_PLOTS, show_plots=SHOW_PLOTS,
                        #           title="New Main Path",
                        #           path=f'{scenario_folder}/Iter_{i + 1}/Plan_{planner.plan_count}')
                        prev_plan_time = time.time()
                        total_cc += common.map_utils.cc_calls

                    states_of_action = []
                    planner.env.set_state(curr_state)
                    obstacles_in_way = -1
                    action_time_count = 0
                    total_action_time_count = 0
                    done = False
                    while (action_idx < main_actions.shape[0] and obstacles_in_way < 0 and not done
                           and (time.time() - prev_plan_time) < 2) and total_action_time_count < 2:
                        curr_state_temp, done, _, visited_states = planner.propagate_action_sequence_env(
                            curr_state, main_actions[action_idx, np.newaxis]
                        )
                        if done is None:
                            print('\033[95m', 10 * '@', "!!!COLLISION DETECTED!!!", 10 * '@', '\033[00m')
                            print(f"Current action {action_idx} / {main_actions.shape[0]}")
                            # plot_traj(planner, scanned_maze, executed_path, block=BLOCK_PLOTS,
                            #           show_plots=SHOW_PLOTS,
                            #           title="Executed Path - Collision",
                            #           path=f'{scenario_folder}/Iter_{i + 1}/Collision')
                            break
                        if done:
                            print('\033[95m', 10 * '-', "!!! REACHED GOAL WOOHOO !!!", 10 * '-', '\033[00m')
                            # plot_traj(planner, scanned_maze, executed_path, block=BLOCK_PLOTS,
                            #           show_plots=SHOW_PLOTS,
                            #           title="Executed Path - Reached Goal",
                            #           path=f'{scenario_folder}/Iter_{i + 1}/ReachedGoal')

                        states_of_action.append(visited_states[0, 1, :])
                        executed_path.append(visited_states[0, 1, :])
                        executed_actions.append(main_actions[action_idx])
                        curr_state = curr_state_temp
                        planner.env.set_state(curr_state)
                        action_idx += 1
                        action_time_count += planner.env.dt
                        total_action_time_count += planner.env.dt
                        if action_time_count > planner.env.lidar2dsim.scan_time:
                            # print("Scanning area...")
                            scan_and_update_maze(planner, maze_data, maze_data_with_obstacle, scanned_maze)
                            # print("Check path crossing obstacles OR actions are done...")
                            obstacles_in_way = check_no_obstacles_in_path(planner, scanned_maze, main_path_array)
                            action_time_count = 0

                    if done is None:
                        break

                    if action_idx < main_actions.shape[0]:
                        print('\033[97m Found obstacle in main path!\033[00m')
                    print(f"Current action {action_idx} / {main_actions.shape[0]}")
                    # plot_traj(planner, scanned_maze, executed_path, block=BLOCK_PLOTS, show_plots=SHOW_PLOTS,
                    #           title='Robot Path until now',
                    #           path=f'{root_folder}/{current_run}/{scenario_name}/Iter_{i + 1}/RobotStatus_{planner.plan_count}')

            # Collect statistics
            end_time = time.time()
            runtime = end_time - start_time
            num_states_in_tree = stats_2_keep["number_of_nodes"]
            num_iterations = stats_2_keep["iterations"]

            plot_traj(planner, scanned_maze, executed_path,
                      block=BLOCK_PLOTS, show_plots=SHOW_PLOTS,
                      title="Executed Path",
                      path=f'{root_folder}/{current_run}/{scenario_name}/{i + 1}')
            if executed_path is not None:
                np.savetxt(
                    os.path.join(planner.save_path, planner.scenario_iter_folder_name, f'path_DP_{i}.csv'),
                    executed_path, fmt='%.6f', delimiter=',')
                success = done == True
                trajectory_time = len(executed_path) * planner.env.dt
                try:
                    trajectory_length = calculate_trajectory_length(np.vstack(executed_path))
                    avg_velocity = calculate_average_velocity(np.vstack(executed_path))
                    ctrl_effort = np.linalg.norm(np.vstack(executed_actions), axis=1)
                    ctrl_effort_max = np.max(ctrl_effort)
                    ctrl_effort_mean = np.mean(ctrl_effort)
                    ctrl_effort_std = np.std(ctrl_effort)
                except Exception as ex:
                    success = -1
                    trajectory_length = -1
                    avg_velocity = -1
                    trajectory_time = -1
                    num_states_in_tree = -1
                    print(
                        f"Error calculating trajectory length for scenario {scenario_name}, planner {planner_name}, run {i}")
                    print(executed_path)
                    print(ex)
                    trajectory_length = 0  # calculate_trajectory_length(executed_path)
                    avg_velocity = 0  #calculate_average_velocity(executed_path)
                    ctrl_effort = 0
                    ctrl_effort_max = 0
                    ctrl_effort_mean = 0
                    ctrl_effort_std = 0
                    # return
            else:
                success = 0
                trajectory_length = -1
                avg_velocity = -1
                trajectory_time = 0
                num_states_in_tree = -1
                ctrl_effort_max = -1
                ctrl_effort_mean = -1
                ctrl_effort_std = -1
            print(f"Avg collision checking calls: {total_cc / (i + 1)} for {i + 1} iterations")
            # Write results to CSV
            with open(scenario_output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [i + 1, success, runtime, trajectory_length, trajectory_time, avg_velocity,
                     num_states_in_tree, num_iterations, ctrl_effort_max, ctrl_effort_mean,
                     ctrl_effort_std])


def calculate_trajectory_length(executed_path):
    return np.sum(np.linalg.norm(np.diff(executed_path[:, :2], axis=0), axis=1))


def calculate_average_velocity(executed_path):
    velocities = np.sqrt(np.square(executed_path[:, 2]) + np.square(executed_path[:, 3]))
    return np.mean(velocities)


if __name__ == "__main__":
    # Carmaze
    diffusion_sampler_checkpoints = {'resnet': 'carmaze_step_40000-001.pt'}
    root_folder = 'benchmark_results/ditree'

    scenario_file = 'experiments/debug1_test_scenarios_car.csv'
    online_time_budget = 2
    for cfg_file in ['carmaze']:
        # col, row
        # 15, 10    [1, 4]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[10, 15], obstacle_dims=[1, 4],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
        # 16, 9     [2, 2]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[9, 16], obstacle_dims=[2, 2],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
        # 16, 7     [2, 3]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[7, 16], obstacle_dims=[2, 3],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
    scenario_file = 'experiments/debug2_test_scenarios_car.csv'
    for cfg_file in ['carmaze']:
        # 7 , 1     [4, 4]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[1, 7], obstacle_dims=[4, 4],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)

    online_time_budget = 4
    scenario_file = 'experiments/debug1_test_scenarios_car.csv'
    for cfg_file in ['carmaze']:
        # col, row
        # 15, 10    [1, 4]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[10, 15], obstacle_dims=[1, 4],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
        # 16, 9     [2, 2]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[9, 16], obstacle_dims=[2, 2],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
        # 16, 7     [2, 3]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[7, 16], obstacle_dims=[2, 3],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
    scenario_file = 'experiments/debug2_test_scenarios_car.csv'
    for cfg_file in ['carmaze']:
        # 7 , 1     [4, 4]
        evaluate_all_scenarios('maps/mazes', scenario_file,
                               obstacle_loc_row_col=[1, 7], obstacle_dims=[4, 4],
                               cfg_file=cfg_file, total_runs=25, offline_time_budget=60,
                               online_time_budget=online_time_budget,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)
    playsound(r'C:\Users\Jonathan\Music\Jon Bellion - All Time Low (Official Music Video).mp3')
