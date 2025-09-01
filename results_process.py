import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

CSV_NAME = 'diffusion_RRT_PD64_carmaze.csv'
WANTED_TECHNIQUES = ['mppi', 'ditree']


def save_stats(technique_results_path):
    res_tot = dict()
    for run_fold in os.listdir(technique_results_path):
        if not os.path.isdir(os.path.join(technique_results_path, run_fold)):
            continue
        run_path = os.path.join(technique_results_path, run_fold)
        items = os.listdir(run_path)
        for item in items:
            if os.path.isdir(os.path.join(run_path, item)):
                scenario_res_path = os.path.join(run_path, item, CSV_NAME)
                df = pd.read_csv(scenario_res_path)
                df_success = df[df['success'] == True]
                res_tot[f'{run_fold}'] = dict()
                res_tot[f'{run_fold}']['success_rate'] = df['success'].sum() / len(df)

                res_tot[f'{run_fold}']['trajectory_len_mean'] = df['trajectory_length'].mean()
                res_tot[f'{run_fold}']['trajectory_len_median'] = df['trajectory_length'].median()
                res_tot[f'{run_fold}']['trajectory_len_std'] = df['trajectory_length'].std()
                res_tot[f'{run_fold}']['trajectory_len_min'] = df['trajectory_length'].min()
                res_tot[f'{run_fold}']['trajectory_len_max'] = df['trajectory_length'].max()

                res_tot[f'{run_fold}']['trajectory_time_mean'] = df['trajectory_time'].mean()
                res_tot[f'{run_fold}']['trajectory_time_median'] = df['trajectory_time'].median()
                res_tot[f'{run_fold}']['trajectory_time_std'] = df['trajectory_time'].std()
                res_tot[f'{run_fold}']['trajectory_time_min'] = df['trajectory_time'].min()
                res_tot[f'{run_fold}']['trajectory_time_max'] = df['trajectory_time'].max()

                res_tot[f'{run_fold}']['runtime_sec_mean'] = df['runtime [sec]'].mean()
                res_tot[f'{run_fold}']['runtime_sec_median'] = df['runtime [sec]'].median()
                res_tot[f'{run_fold}']['runtime_sec_std'] = df['runtime [sec]'].std()
                res_tot[f'{run_fold}']['runtime_sec_min'] = df['runtime [sec]'].min()
                res_tot[f'{run_fold}']['runtime_sec_max'] = df['runtime [sec]'].max()

                res_tot[f'{run_fold}']['time_ratio_mean'] = (df['trajectory_time'] / df['runtime [sec]']).mean()
                res_tot[f'{run_fold}']['time_ratio_median'] = (df['trajectory_time'] / df['runtime [sec]']).median()
                res_tot[f'{run_fold}']['time_ratio_std'] = (df['trajectory_time'] / df['runtime [sec]']).std()
                res_tot[f'{run_fold}']['time_ratio_min'] = (df['trajectory_time'] / df['runtime [sec]']).min()
                res_tot[f'{run_fold}']['time_ratio_max'] = (df['trajectory_time'] / df['runtime [sec]']).max()

                res_tot[f'{run_fold}']['num_iterations_mean'] = (df['num_RRT_iterations']).mean()
                res_tot[f'{run_fold}']['num_iterations_median'] = (df['num_RRT_iterations']).median()
                res_tot[f'{run_fold}']['num_iterations_std'] = (df['num_RRT_iterations']).std()
                res_tot[f'{run_fold}']['num_iterations_min'] = (df['num_RRT_iterations']).min()
                res_tot[f'{run_fold}']['num_iterations_max'] = (df['num_RRT_iterations']).max()

                res_tot[f'{run_fold}']['OUT_OF_SUCCESS'] = 0

                res_tot[f'{run_fold}']['s_trajectory_len_mean'] = df_success['trajectory_length'].mean()
                res_tot[f'{run_fold}']['s_trajectory_len_median'] = df_success['trajectory_length'].median()
                res_tot[f'{run_fold}']['s_trajectory_len_std'] = df_success['trajectory_length'].std()
                res_tot[f'{run_fold}']['s_trajectory_len_min'] = df_success['trajectory_length'].min()
                res_tot[f'{run_fold}']['s_trajectory_len_max'] = df_success['trajectory_length'].max()

                res_tot[f'{run_fold}']['s_trajectory_time_mean'] = df_success['trajectory_time'].mean()
                res_tot[f'{run_fold}']['s_trajectory_time_median'] = df_success['trajectory_time'].median()
                res_tot[f'{run_fold}']['s_trajectory_time_std'] = df_success['trajectory_time'].std()
                res_tot[f'{run_fold}']['s_trajectory_time_min'] = df_success['trajectory_time'].min()
                res_tot[f'{run_fold}']['s_trajectory_time_max'] = df_success['trajectory_time'].max()

                res_tot[f'{run_fold}']['s_runtime_sec_mean'] = df_success['runtime [sec]'].mean()
                res_tot[f'{run_fold}']['s_runtime_sec_median'] = df_success['runtime [sec]'].median()
                res_tot[f'{run_fold}']['s_runtime_sec_std'] = df_success['runtime [sec]'].std()
                res_tot[f'{run_fold}']['s_runtime_sec_min'] = df_success['runtime [sec]'].min()
                res_tot[f'{run_fold}']['s_runtime_sec_max'] = df_success['runtime [sec]'].max()

                res_tot[f'{run_fold}']['s_time_ratio_mean'] = (
                        df_success['trajectory_time'] / df_success['runtime [sec]']).mean()
                res_tot[f'{run_fold}']['s_time_ratio_median'] = (
                        df_success['trajectory_time'] / df_success['runtime [sec]']).median()
                res_tot[f'{run_fold}']['s_time_ratio_std'] = (
                        df_success['trajectory_time'] / df_success['runtime [sec]']).std()
                res_tot[f'{run_fold}']['s_time_ratio_min'] = (
                        df_success['trajectory_time'] / df_success['runtime [sec]']).min()
                res_tot[f'{run_fold}']['s_time_ratio_max'] = (
                        df_success['trajectory_time'] / df_success['runtime [sec]']).max()

                res_tot[f'{run_fold}']['s_num_iterations_mean'] = (df_success['num_RRT_iterations']).mean()
                res_tot[f'{run_fold}']['s_num_iterations_median'] = (df_success['num_RRT_iterations']).median()
                res_tot[f'{run_fold}']['s_num_iterations_std'] = (df_success['num_RRT_iterations']).std()
                res_tot[f'{run_fold}']['s_num_iterations_min'] = (df_success['num_RRT_iterations']).min()
                res_tot[f'{run_fold}']['s_num_iterations_max'] = (df_success['num_RRT_iterations']).max()

    res_df = pd.DataFrame.from_dict(res_tot)
    res_df.to_csv(os.path.join(results_dir, f'Results_{technique}.csv'))


def plot_scenario_comparison(df_list_all_algos, sampling_names, algo_names, scenario_name="Scenario"):
    """
    Compare multiple algorithms across multiple sampling techniques for a given scenario,
    with clean academic-style plots.

    Parameters
    ----------
    df_list_all_algos : list of list of pd.DataFrame
        Each element is a list of dataframes for a single algorithm
        (one dataframe per sampling technique).
        Example: [[df_algo1_tech1, df_algo1_tech2], [df_algo2_tech1, df_algo2_tech2], ...]

    sampling_names : list of str
        Names of the sampling techniques (same order for all algorithms).

    algo_names : list of str
        Names of the algorithms.

    scenario_name : str
        Scenario name for figure titles.
    """

    # Clean academic theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0, palette="Set2")

    # Merge all algorithms into one DataFrame
    df_all = pd.concat(
        [
            pd.concat(
                [df.assign(sampling=tech, algorithm=algo_name) for df, tech in zip(df_list_algo, sampling_names)],
                ignore_index=True
            )
            for df_list_algo, algo_name in zip(df_list_all_algos, algo_names)
        ],
        ignore_index=True
    )

    # ---- 1. Success Rate ----
    success = df_all.groupby(["sampling", "algorithm"])["success"].mean().reset_index()
    success["success_rate"] = success["success"] * 100

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=success, x="sampling", y="success_rate", hue="algorithm",
        edgecolor="black", linewidth=0.8
    )
    ax.set_ylabel("Success Rate [%]")
    ax.set_xlabel("Sampling Technique")
    ax.set_title(f"{scenario_name} - Success Rate")
    plt.xticks(rotation=20)
    plt.legend(title="Algorithm", frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # ---- Boxplots for continuous metrics ----
    metrics = [
        ("runtime [sec]", "Runtime [s]"),
        ("trajectory_length", "Trajectory Length"),
        ("trajectory_time", "Trajectory Time [s]"),
        ("num_RRT_iterations", "# RRT Iterations"),
        ("ctrl_effort_max", "Control Effort (Max)"),
        ("ctrl_effort_mean", "Control Effort (Mean)"),
        ("ctrl_effort_std", "Control Effort (Std)"),
    ]

    for col, ylabel in metrics:
        plt.figure(figsize=(9, 6))
        ax = sns.boxplot(
            data=df_all, x="sampling", y=col, hue="algorithm",
            showcaps=True, whiskerprops={'color': 'black'},
            medianprops={'color': 'black', 'linewidth': 1.2}
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sampling Technique")
        ax.set_title(f"{scenario_name} - {ylabel}")
        plt.xticks(rotation=20)
        plt.legend(title="Algorithm", frameon=False, loc='upper right')
        sns.despine()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # results_dir = 'benchmark_results'
    # for technique in os.listdir(results_dir):
    #     if technique not in WANTED_TECHNIQUES:
    #         continue
    #     technique_results_path = os.path.join(results_dir, technique)
    #     save_stats(technique_results_path)
        # save_plots(technique_results_path)

    results_dir_final = 'benchmark_results_final/DiTree'
    folders_list = os.listdir(results_dir_final)
    sorted_algs = [
        (int(s[0]), int(s[2]), os.path.join(results_dir_final, s), os.listdir(os.path.join(results_dir_final, s)))
        for s in folders_list
    ]

    # create plots for algorithm with 2 seconds planning online
    alg_2_seconds_algs = sorted([s for s in sorted_algs if s[1] == 2], key=lambda x: x[0])
    alg_4_seconds_algs = sorted([s for s in sorted_algs if s[1] == 4], key=lambda x: x[0])
    sampling_technique_names = ['Original',     # Original
                                'Sampling 1',   # Original + Sampling initial path
                                'Sampling 2',   # Prior + Sampling initial path
                                'Sampling 3',   # Prior + Gaussian + Sampling initial path
                                'Sampling 4']   # Prior + Gaussian + Sampling initial path + Planning every 2 seconds
    scenario_names = [
        'Boxes Demo - 1 by 4 obstacle',
        'Boxes Demo - 2 by 2 obstacle',
        'Boxes Demo - 2 by 3 obstacle',
        'Boxes - 4 by 4 obstacle',
    ]
    for scen_idx in range(4):
        print(10 * '*', f'Creating plots for Scenario Number {scen_idx + 1}', 10 * '*')
        df_list_2secs = []
        df_list_4secs = []
        for alg_idx, _, path, res_folds in alg_2_seconds_algs:
            scenarios_list = sorted(int(i) for i in os.listdir(path))
            scenario_path = os.path.join(path, str(scenarios_list[scen_idx]))
            maze_name = os.listdir(scenario_path)[0]
            scenario_path = os.path.join(scenario_path, maze_name, CSV_NAME)
            print('\t', scenario_path)
            df_list_2secs.append(pd.read_csv(scenario_path))
        for alg_idx, _, path, res_folds in alg_4_seconds_algs:
            scenarios_list = sorted(int(i) for i in os.listdir(path))
            scenario_path = os.path.join(path, str(scenarios_list[scen_idx]))
            maze_name = os.listdir(scenario_path)[0]
            scenario_path = os.path.join(scenario_path, maze_name, CSV_NAME)
            print('\t', scenario_path)
            df_list_4secs.append(pd.read_csv(scenario_path))

        plot_scenario_comparison(df_list_all_algos=[df_list_2secs, df_list_4secs],
                                 sampling_names=sampling_technique_names,
                                 algo_names=("2 Seconds Online Planning",
                                             "4 Seconds Online Planning",
                                             "MPPI(T=8, K=20)",
                                             "MPPI(T=16, K=10)"),
                                 scenario_name=scenario_names[scen_idx])
