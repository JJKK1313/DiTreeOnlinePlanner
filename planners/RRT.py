import os
import random
import time

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.spatial import KDTree

from planners.base_planner import Node, BasePlanner
from common.map_utils import create_local_map
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.fm_policy import DiffusionSampler


class RRT_Planner(BasePlanner):
    def __init__(self, start_state, goal_state, environment, sampler, **kwargs):
        super().__init__(start_state, goal_state, environment, sampler, **kwargs)

        self.kd_tree_dim = 2
        self.kd_tree = KDTree([start_state[:self.kd_tree_dim]])
        self.goal_sample_rate = 0.15
        self.goal_conditioning_bias = kwargs.get("goal_conditioning_bias", 0.85)  # default is 0.85
        self.prop_duration_schedule = kwargs.get("prop_duration", [64])  # default is [256,128,64] | [64]
        self.offline_time_budget = kwargs.get("offline_time_budget", 60)
        self.plan_count = 0
        self.init_main_path = None

    def reset(self, start_state: np.ndarray = None, goal_state: np.ndarray = None, reset_main_path: bool = False):
        if reset_main_path:
            self.init_main_path = None
        if start_state is not None:
            self.start_node = Node(start_state)
            self.goal_state = goal_state
            self.options["reset_cell"] = self.env.cell_xy_to_rowcol(start_state[:2])
            self.options["reset_deg"] = np.rad2deg(start_state[2])
            self.options["goal_cell"] = self.env.cell_xy_to_rowcol(goal_state[:2])
        self.node_list.clear()
        self.node_list = [self.start_node]
        self.failed_node_list = []
        self.kd_tree = KDTree([self.start_node.state[:self.kd_tree_dim]])
        self.results = {"iterations": 0, "time": 0, "path": None, "actions": None, "number_of_nodes": 0}
        self.env.reset(options=self.options)

    def nearest_node(self, sample):
        _, index = self.kd_tree.query(sample[:, :self.kd_tree_dim], k=1)
        return self.node_list[index[0]]

    def nearest_node_batch(self, samples):
        _, index = self.kd_tree.query(samples)
        return [self.node_list[i] for i in index]

    def update_maze(self, new_maze):
        self.maze = new_maze
        self.env.maze_map = new_maze

    def check_obstacle_ahead(self, state):
        x, y, theta = state[:3]
        row, col = self.env.cell_xy_to_rowcol([x, y], floor_enable=False)
        # print(row, col, theta)
        samples = np.linspace(0, 1.5, 30)  # sqrt(2) for at least two diagonals of units.
        samples = samples[np.newaxis].T @ np.array([[np.cos(-theta), np.sin(-theta)]])
        samples = samples + np.array([col, row])
        samples_q = samples.astype('int')
        samples_q = np.clip(samples_q, [0, 0], np.array(self.maze.shape[::-1]) - 1)
        # plt.figure()
        # plt.imshow(self.maze, origin='lower', extent=[0, self.maze.shape[0], 0, self.maze.shape[1]])
        # plt.xticks(np.arange(0, self.maze.shape[1], 1))
        # plt.yticks(np.arange(0, self.maze.shape[0], 1))
        # plt.grid()
        # plt.scatter(samples_q[:, 0], samples_q[:, 1])   # columns are x and rows are y
        # plt.scatter(samples[:, 0], samples[:, 1])
        # plt.scatter(samples[0, 0], samples[0, 1], color='r')
        # plt.show()
        if np.any(self.maze[samples_q[:, 1], samples_q[:, 0]]):
            return True
        return False

    def extract_path_after_obstacle(self):
        main_path_array = self.init_main_path[:, :2].copy()
        # remove the rest of the path as it is no more relevant.
        curr_state = self.env.state[:2]
        closest_node_on_path_idx = np.argmin(np.linalg.norm(curr_state - self.init_main_path[:, :2], axis=1))
        main_path_array = main_path_array[closest_node_on_path_idx:, :]
        # find where the path is crossing an obstacle
        main_path_points_rowcol = np.array(
            list(map(lambda x: self.env.cell_xy_to_rowcol(x).astype('int'), main_path_array)))
        obstacle_in_path_loc_idx = -1
        for idx, p in enumerate(main_path_points_rowcol):
            if self.maze[p[0], p[1]] == 1:
                obstacle_in_path_loc_idx = idx
                break
        cp = main_path_points_rowcol[obstacle_in_path_loc_idx]
        while self.maze[cp[0], cp[1]] == 1:
            obstacle_in_path_loc_idx += 1
            cp = main_path_points_rowcol[obstacle_in_path_loc_idx]
        # plt.figure()
        # plt.imshow(self.maze, origin='lower', extent=[0, self.maze.shape[0], 0, self.maze.shape[1]])
        # for i, s in enumerate(main_path_array[obstacle_in_path_loc_idx:]):
        #     s = s.copy()
        #     s[:2] = self.env.cell_xy_to_rowcol(s[:2], floor_enable=False)
        #     s[:2] = s[:2][::-1]
        #     plt.scatter(s[0], s[1], s=5, color='#ff0000' if self.maze[int(s[1]), int(s[0])] else '#00ff00')
        # cu_rc = self.env.cell_xy_to_rowcol(curr_state, floor_enable=False)[::-1]
        # plt.scatter(cu_rc[0], cu_rc[1], s=100)
        # plt.show()
        return main_path_array[obstacle_in_path_loc_idx:]

    def plan(self):
        """
        This function will plan as long as it has left time to do so (under the time budget)
        Returns: a path plan.
        """
        start_time = time.time()
        curr_time = time.time()
        total_diffusion_time = 0
        iter_num = 0
        orig_prob_map = self.env.prob_map.copy()
        has_obstacle_ahead = []
        self.env.update_prob_map_by_loc()
        remain_init_path = self.extract_path_after_obstacle() if self.init_main_path is not None else None
        sample_node = [None]
        while (curr_time - start_time) < self.time_budget:
            if self.verbose:
                print(f"\rIteration: {iter_num}, Elapsed Time: {(curr_time - start_time):.2f} seconds", end="")
            if remain_init_path is not None:
                node_idx = np.random.choice(np.arange(len(remain_init_path)))
                # 40% chance for exploration over exploitation
                sample_node = self.random_node_sample() if random.random() < 0.4 else remain_init_path[node_idx][
                    np.newaxis]
            else:
                sample_node = self.random_node_sample()  # sample a random state.

            curr_node = self.nearest_node(sample_node)
            curr_state = curr_node.state
            full_action_seq = None
            full_states_seq = None
            done = False
            prev_actions = curr_node.parent_action_seq
            prev_states = curr_state[None, None, :] if curr_node.parent_states_seq is None \
                else curr_node.parent_states_seq  # (1,1,obs_dim)
            edge_length = self.prop_duration_schedule[
                np.clip(curr_node.num_visit, 0, len(self.prop_duration_schedule) - 1)
            ]
            curr_node.num_visit += 1
            goal = sample_node[0, :2] # if random.random() > self.goal_conditioning_bias else self.goal_state[:2]
            for j in range(edge_length // self.action_horizon):
                iter_num += 1
                yaw = curr_state[2]
                local_map = create_local_map(self.maze, curr_state[0],
                                             curr_state[1], yaw,
                                             self.local_map_size,
                                             self.local_map_scale, self.s_global,
                                             (self.x_center, self.y_center))

                local_map = torch.tensor(local_map).to(self.device)

                # Sample and trim the action sequence
                start_diffusion_time = time.time()
                sampled_actions = self.sampler(prev_states,
                                               prev_actions=prev_actions,
                                               goal=goal,
                                               local_map=local_map)[0, :self.action_horizon]
                end_diffusion_time = time.time()
                total_diffusion_time += (end_diffusion_time - start_diffusion_time)

                curr_state, done, curr_action_seq, curr_states_seq = self.propagate_action_sequence_env(curr_state,
                                                                                                        sampled_actions)
                if done is None:  # Collision
                    if self.save_bad_edges:
                        self.failed_node_list.append(Node(curr_state, full_action_seq, full_states_seq,
                                                          parent=curr_node))
                    curr_state = None
                    break
                # Fill edge's state/action sequence
                full_action_seq = np.concatenate((full_action_seq, curr_action_seq)) \
                    if full_action_seq is not None else curr_action_seq
                prev_actions = curr_action_seq
                full_states_seq = np.concatenate((full_states_seq, curr_states_seq), axis=1) \
                    if full_states_seq is not None else curr_states_seq
                prev_states = curr_states_seq
                if done:
                    break

            if curr_state is not None and done is not None:
                non_zero_rows_mask = ~(full_action_seq == 0).all(axis=1)
                full_action_seq = full_action_seq[non_zero_rows_mask]
                non_zero_rows_mask = ~(full_states_seq[0] == 0).all(axis=1)
                full_states_seq = full_states_seq[0, non_zero_rows_mask][np.newaxis]
                new_node = Node(curr_state, full_action_seq, full_states_seq, parent=curr_node)
                self.node_list.append(new_node)
                has_obstacle_ahead.append(self.check_obstacle_ahead(curr_state))
                self.kd_tree = KDTree([node.state[:self.kd_tree_dim] for node in self.node_list])
                if done:
                    # print diffusion time and total time
                    curr_time = time.time()
                    if self.verbose:
                        print(f"\rIteration: {iter_num}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                              f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
                    self.env.prob_map = orig_prob_map
                    # self.visualize_tree(filename=f'plan_{self.plan_count}_iter_{iter_num}', debug=self._debug,
                    #                     goal_state=sample_node[0], has_obs_ahead_list=has_obstacle_ahead.copy())
                    return self.handle_goal_reached(new_node, iter_num, start_time)

            curr_time = time.time()
        curr_time = time.time()
        if self.verbose:
            print(f"\rIteration: {iter_num}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                  f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
        if np.all(has_obstacle_ahead):
            # self.visualize_tree(filename=f'plan_{self.plan_count}_iter_{iter_num}', debug=self._debug,
            #                     has_obs_ahead_list=has_obstacle_ahead.copy())
            self.env.prob_map = orig_prob_map
            self.results["iterations"] = iter_num
            self.results["number_of_nodes"] = len(self.node_list)
            return None, None

        node_list_wo_start = self.node_list[1:]
        if self.init_main_path is None:
            distances = np.array([np.linalg.norm(nd.state[:2] - self.goal_state[:2]) for nd in node_list_wo_start])
            cost = distances + 10e3 * np.array(has_obstacle_ahead, dtype='int')
            nearest_sample_to_goal = node_list_wo_start[np.argmin(cost)]
        else:
            # calc for each new node list candidate the nearest neighbor index along main path
            nn_index_along_main_path = \
                np.array(
                    [  # list comprehension
                        np.argmin(np.linalg.norm(nd.state[:2] - self.init_main_path[:, :2], axis=1))
                        if not has_obstacle_ahead[idx] else -1 for idx, nd in enumerate(node_list_wo_start)
                    ]
                )
            # if it has obstacle then add big cost
            nearest_sample_to_goal = node_list_wo_start[np.argmax(nn_index_along_main_path)]
        self.env.prob_map = orig_prob_map
        # self.visualize_tree(filename=f'plan_{self.plan_count}_iter_{iter_num}', debug=self._debug,
        #                     goal_state=sample_node[0], has_obs_ahead_list=has_obstacle_ahead.copy())
        return self.handle_goal_reached(nearest_sample_to_goal, iter_num, start_time)


if __name__ == "__main__":
    # Settings
    debug = True
    time_budget = 120

    seed = 42
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = 'pointmaze-medium-v2'
    checkpoint = 'pointmaze_200.pt'
    obs_horizon = 1
    num_diffusion_iters = 100
    if env_id == "antmaze-large-diverse-v1":
        obs_dim = 27
        action_dim = 8
    elif env_id == "pointmaze-medium-v2":
        obs_dim = 4
        action_dim = 2
    elif 'drone' in env_id:
        obs_dim = 10
        action_dim = 4

    dataset = minari.load_dataset(env_id, download=False)
    render_mode = 'human' if debug else 'rgb_array'
    env = dataset.recover_environment(render_mode=render_mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'checkpoints/'
    checkpoint = torch.load(output_dir + checkpoint)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    noise_pred_net = noise_pred_net.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    episode = dataset[2]
    start = episode.observations['observation'][0]
    start[2:] = 0  # start stationary
    goal = episode.observations['desired_goal'][1]

    diffusion_sampler = DiffusionSampler(noise_pred_net, noise_scheduler, env_id,
                                         pred_horizon=16,
                                         action_dim=2,
                                         obs_history=1,
                                         goal_conditioned=False
                                         )

    diffusion_planner = RRT_planner(start, goal,
                                    env_id=env_id,
                                    environment=env,
                                    sampler=diffusion_sampler,
                                    time_budget=time_budget,
                                    max_iter=300,
                                    verbose=True,
                                    render=True,
                                    )
    print("Planning with Diffusion Sampler...")
    path_diffusion, actions_diffusion = diffusion_planner.plan()

    # kinoRRT = RRT_planner(start, goal,
    #                       env_id=env_id,  # 'pushT' or 'maze'
    #                       environment=env,
    #                       sampler=UniformSampler(env.action_space),
    #                       time_budget=time_budget,
    #                       max_iter=10000,
    #                       bounds=None  # environment bounds
    #                       )
    # print("Planning with Kino RRT...")
    # path_kino, actions_kino = kinoRRT.plan()

    # car_params = {
    #     'L': 2.0,  # Wheelbase
    #     'max_speed': 1.0  # Maximum speed
    # }

    # rrt = RRT(start, goal, car_params)
    # path = rrt.plan()

    # if path is not None:
    #     rrt.visualize(path)
    # else:
    #     print("Path not found.")
