import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patheffects as path_effects

CSV_NAME = 'diffusion_RRT_PD64_carmaze.csv'
WANTED_TECHNIQUES = ['mppi', 'ditree']
PALETTE = sns.color_palette()

MEAN_PROPS = dict(linestyle='--', color='black')
SHOWFLIERS = True
ORDER = ['Original', 'Original+Ref', 'OM+Ref', 'OM+LB+Ref', 'All+ForceReplan']


def add_median_labels(ax: plt.Axes, fmt: str = ".2f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


# def plot_results(df):
#     """
#     df columns expected:
#         - algorithm
#         - sampling_tech
#         - scenario
#         - params
#         - success
#         - iterations
#         - runtime
#         - ctrl_mean
#     """
#
#     df = df.copy()
#     # Build unified config label
#     df["Config"] = df.apply(
#         lambda row: (
#             f"MPPI({row['params']})"
#             if row["algorithm"] == "MPPI"
#             else f"DiTree-{row['sampling_tech']}({row['params']})"
#         ),
#         axis=1
#     )
#
#     sns.set(style="whitegrid", context="paper", font_scale=1.1)
#
#     # --- 1. Summary bar plots (success rate only) ---
#     scenarios = df["scenario"].unique()
#     n_scenarios = len(scenarios)
#
#     fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True, constrained_layout=True)
#     plt.subplots_adjust(top=0.95, bottom=0.08, left=0.25, right=0.95, hspace=0.4)
#
#     if n_scenarios == 1:
#         axes = [axes]
#
#     for ax, scen in zip(axes, scenarios):
#         scen_df = df[df["scenario"] == scen]
#         sns.barplot(
#             data=scen_df, x="success", y="Config", ax=ax,
#             errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
#         )
#         for p in ax.patches:
#             width = p.get_width()  # horizontal bar width
#             ax.text(
#                 x=width + 0.01,  # slightly past the end of the bar
#                 y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
#                 s=f"{width * 100:.2f}%",  # percentage text
#                 ha='left',  # align left to avoid overlap
#                 va='center'  # vertically centered
#             )
#         ax.set_title(f"Scenario: {scen}", fontsize=13, pad=5)
#         ax.set_xlabel("")
#         ax.set_ylabel("Success Rate" if ax == axes[0] else "")
#         ax.tick_params(axis="x")
#         ax.margins(y=0.01)
#
#     plt.tight_layout()
#     plt.show()
#
#     # --- 2. Detailed per-scenario plots ---
#     metrics = {
#         "Success Rate": "success",
#         "Iteration Count": "num_RRT_iterations",
#         "Trajectory Length": "trajectory_length",
#         "Runtime": "runtime [sec]",
#         # "Control Mean": "ctrl_effort_mean"
#     }
#
#     for scen in scenarios:
#         scen_df = df[df["scenario"] == scen]
#
#         fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False, constrained_layout=True)
#         fig.suptitle(f"Scenario: {scen}", fontsize=15)
#
#         for ax, (title, col) in zip(axes, metrics.items()):
#             if title == "Success Rate":
#                 sns.barplot(
#                     data=scen_df, x=col, y="Config", ax=ax,
#                     errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
#                 )
#                 for p in ax.patches:
#                     width = p.get_width()  # horizontal bar width
#                     ax.text(
#                         x=width + 0.01,  # slightly past the end of the bar
#                         y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
#                         s=f"{width * 100:.2f}%",  # percentage text
#                         ha='left',  # align left to avoid overlap
#                         va='center'  # vertically centered
#                     )
#             else:
#                 sns.boxplot(
#                     data=scen_df, x=col, y="Config", ax=ax,
#                     palette=PALETTE, hue='sampling_tech', showfliers=False, showmeans=True,  #meanline=True,
#                     meanprops=MEAN_PROPS, patch_artist=True
#                 )
#                 add_median_labels(ax)
#                 # Add text labels for medians
#                 for median in bp['medians']:
#                     x, y = median.get_xdata()[0], median.get_ydata()[0]  # Get the start point of the median line
#                     median_value = median.get_ydata()[0]  # The median value is the y-coordinate
#                     ax.text(x, y, f'{median_value:.2f}', ha='center', va='bottom', color='red', fontsize=10)
#             ax.set_title(title, fontsize=13, pad=5)
#             ax.set_xlabel("")
#             ax.tick_params(axis="x")
#             ax.margins(y=0.01)
#
#         plt.tight_layout()
#         plt.show()


def plot_splitted_summarize_scenarios(df_ditree, df_mppi, metrics: dict = None):
    sns.set(style="whitegrid", context="paper", font_scale=1.1)

    df_ditree = df_ditree.drop('scenario', axis=1)
    df_ditree['success'] *= 100
    df_mppi = df_mppi.drop('scenario', axis=1)
    df_mppi['success'] *= 100
    # --- 1. Summary bar plots (success rate only) ---

    for (title, col) in metrics.items():
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False, constrained_layout=True)
        print(title)
        for ax, df in zip(axes, [df_ditree, df_mppi]):
            if "Success Rate" in title:
                sns.barplot(
                    data=df, x="sampling_tech", y="success", ax=ax, dodge=True,
                    errorbar=None, palette=PALETTE, hue='params', order=ORDER
                )
                ax.legend(loc='best')
                ax.set_ylim([0, 100])
                for p in ax.patches:
                    if p.get_width() == 0:
                        continue
                    height = p.get_height()  # horizontal bar width
                    ax.text(
                        x=p.get_x(),  # slightly past the end of the bar
                        y=height + 2,  # vertical center of the bar
                        s=f"{height:.2f}%",  # percentage text
                        ha='left',  # align left to avoid overlap
                        va='center'  # vertically centered
                    )
            else:
                sns.boxplot(
                    data=df, x="sampling_tech", y=col, ax=ax,
                    hue='params', showfliers=SHOWFLIERS, palette=PALETTE, meanline=True, showmeans=True,
                    meanprops=MEAN_PROPS, order=ORDER
                )
                add_median_labels(ax)

            ax.set_title(f"{df['algorithm'].iloc[0]} Results", fontsize=12)
            ax.set_ylabel(f"{title}")
            ax.set_xlabel("")
            ax.tick_params(axis="x")
            ax.margins(y=0.01)
        plt.suptitle(f"{title} Comparison: DiTree V.S. MPPI", fontsize=20)
        # plt.tight_layout()
        plt.show()


def plot_compare_success_only_scenarios(df, metrics: dict = None):
    sns.set(style="whitegrid", context="paper", font_scale=1.1)

    df = df.drop('scenario', axis=1)
    df_success_only = df[df['success'] == True]
    df['success'] *= 100
    df_success_only['success'] *= 100
    df_success_only['params'] = 'Success ' + df_success_only['params']
    df_full = pd.concat([df, df_success_only])
    # --- 1. Summary bar plots (success rate only) ---

    for (title, col) in metrics.items():
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(1, 1, 1)
        print(title)
        if "Success Rate" in title:
            sns.barplot(
                data=df, x="sampling_tech", y="success", ax=ax, dodge=True,
                errorbar=None, palette=PALETTE, hue='params', order=ORDER
            )
            ax.legend(loc='best')
            ax.set_ylim([0, 100])
            for p in ax.patches:
                if p.get_width() == 0:
                    continue
                height = p.get_height()  # horizontal bar width
                ax.text(
                    x=p.get_x(),  # slightly past the end of the bar
                    y=height + 2,  # vertical center of the bar
                    s=f"{height:.2f}%",  # percentage text
                    ha='left',  # align left to avoid overlap
                    va='center'  # vertically centered
                )
        else:
            sns.boxplot(
                data=df_full, x="sampling_tech", y=col, ax=ax,
                hue='params', showfliers=SHOWFLIERS, palette=PALETTE, meanline=True, showmeans=True,
                meanprops=MEAN_PROPS, order=ORDER
            )
            add_median_labels(ax)

        ax.set_title(f"{df['algorithm'].iloc[0]} Results", fontsize=12)
        ax.set_ylabel(f"{title}")
        ax.set_xlabel("")
        ax.tick_params(axis="x")
        ax.margins(y=0.01)
        plt.suptitle(f"{title} Comparison to only Successful Runs", fontsize=20)
        # plt.tight_layout()
        plt.show()


def plot_results_summarize_scenarios(df, metrics: dict = None, successful_only=False):
    df["Config"] = df.apply(
        lambda row: (
            f"MPPI({row['params']})"
            if row["algorithm"] == "MPPI"
            else f"DiTree-{row['sampling_tech']}({row['params']})"
        ),
        axis=1
    )
    df['success'] *= 100
    # --- 2. Detailed per-scenario plots ---
    if metrics is None:
        metrics = {
            "Success Rate (%)": "success",
            "Iteration Count": "num_RRT_iterations",
            "Trajectory Length": "trajectory_length",
            "Runtime": "runtime [sec]",
            # "Control Mean": "ctrl_effort_mean"
        }
    scenarios = df["scenario"].unique()

    for scen in scenarios:
        print(scen)
        scen_df = df[df["scenario"] == scen]

        fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=False, constrained_layout=True)
        axes = axes.flatten()
        suptitle = scen if not successful_only else "Succeeded Only: " + scen
        fig.suptitle(scen, fontsize=15)

        for ax, (title, col) in zip(axes, metrics.items()):
            if "Success Rate" in title:
                sns.barplot(
                    data=scen_df, x="sampling_tech", y=col, ax=ax,
                    errorbar=None, palette=PALETTE, hue='params', order=ORDER
                )
                for p in ax.patches:
                    if p.get_width() == 0:
                        continue
                    height = p.get_height()  # horizontal bar width
                    ax.text(
                        x=p.get_x(),  # slightly past the end of the bar
                        y=height + 2,  # vertical center of the bar
                        s=f"{height:.2f}%",  # percentage text
                        ha='left',  # align left to avoid overlap
                        va='center'  # vertically centered
                    )
                ax.set_ylim([0, 100])
            else:
                sns.boxplot(
                    data=scen_df, x="sampling_tech", y=col, ax=ax,
                    palette=PALETTE, hue='params', showfliers=SHOWFLIERS, showmeans=True,  #meanline=True,
                    meanprops=MEAN_PROPS, order=ORDER
                )
                add_median_labels(ax)
                # Add text labels for medians
            ax.set_title(title, fontsize=13, pad=5)
            ax.set_xlabel("")
            ax.tick_params(axis="x")
            ax.margins(y=0.01)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    df = None
    results_dir_final = 'benchmark_results_final'
    for alg in os.listdir(results_dir_final):
        if alg in ['DiTreeOld', 'DiTreeNew', 'DiTreeNew2']:
            continue
        alg_fold = os.path.join(results_dir_final, alg)
        for params in os.listdir(alg_fold):
            params_fold = os.path.join(alg_fold, params)
            for scenario_name in os.listdir(params_fold):
                scenario_fold = os.path.join(params_fold, scenario_name)
                for sampling_tech in os.listdir(scenario_fold):
                    sampling_tech_fold = os.path.join(scenario_fold, sampling_tech)
                    for run_num in os.listdir(sampling_tech_fold):
                        run_path = os.path.join(sampling_tech_fold, run_num)
                        for maze_name in os.listdir(run_path):
                            if os.path.isdir(os.path.join(run_path, maze_name)):
                                csv_path = os.path.join(run_path, maze_name, CSV_NAME)
                                df_temp = pd.read_csv(csv_path).assign(algorithm=alg, params=params,
                                                                       scenario=scenario_name,
                                                                       sampling_tech=sampling_tech)
                                print(len(df_temp), csv_path)
                                df = pd.concat([df, df_temp]) if df is not None else df_temp
    df = df.replace('True', True)
    df = df.replace('False', False)
    df['success'] = df['success'].replace('-1', False)
    df = df.replace('Sampling 1', 'Original+Ref')
    df = df.replace('Sampling 2', 'OM+Ref')
    df = df.replace('CM+Ref', 'OM+Ref')
    df = df.replace('Sampling 3', 'OM+LB+Ref')
    df = df.replace('CM+LB+Ref', 'OM+LB+Ref')
    df = df.replace('Sampling 4', 'All+ForceReplan')
    df = df.replace('All+ConstReplanTime', 'All+ForceReplan')
    df = df.replace("Regular", "Original")
    df = df.replace("T16_K10", "MPPI(K=10, T=16)")
    df = df.replace("T8_K20", "MPPI(K=20, T=8)")
    df = df.replace("2 seconds planning", "2 sec re-plan")
    df = df.replace("4 seconds planning", "4 sec re-plan")
    df['algorithm'] = df['algorithm'].apply(lambda x: 'MPPI' if x == 'MPPI' else 'DiTree')
    df_ditree = df[df['algorithm'] != 'MPPI']
    df_mppi = df[df['algorithm'] == 'MPPI']
    df_mppi['sampling_tech'] = 'Original'

    metrics = {
        "Success Rate (%)": "success",
        "Iteration Count": "num_RRT_iterations",
        "Trajectory Length": "trajectory_length",
        "Runtime": "runtime [sec]",
    }

    plot_compare_success_only_scenarios(df_ditree, metrics)
    plot_splitted_summarize_scenarios(df_ditree.copy(), df_mppi.copy(), metrics)
    plot_results_summarize_scenarios(df.copy())

    df_ditree_success = df_ditree[df_ditree['success'] == True].copy()
    df_mppi_success = df_mppi[df_mppi['success'] == True].copy()
    metrics.pop('Success Rate (%)')

    plot_splitted_summarize_scenarios(df_ditree_success.copy(), df_mppi_success.copy(), metrics)
    plot_results_summarize_scenarios(df[df['success'] == True].copy(), metrics=metrics)

    # plot_results_success_only(df.copy())
    # plot_results(df.copy())
