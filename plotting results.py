import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

CSV_NAME = 'diffusion_RRT_PD64_carmaze.csv'
WANTED_TECHNIQUES = ['mppi', 'ditree']
PALETTE = sns.color_palette()
def plot_results(df):
    """
    df columns expected:
        - algorithm
        - sampling_tech
        - scenario
        - params
        - success
        - iterations
        - runtime
        - ctrl_mean
    """

    df = df.copy()
    # Build unified config label
    df["Config"] = df.apply(
        lambda row: (
            f"MPPI({row['params']})"
            if row["algorithm"] == "MPPI"
            else f"DiTree-{row['sampling_tech']}({row['params']})"
        ),
        axis=1
    )

    sns.set(style="whitegrid", context="paper", font_scale=1.1)

    # --- 1. Summary bar plots (success rate only) ---
    scenarios = df["scenario"].unique()
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True, constrained_layout=True)
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.25, right=0.95, hspace=0.4)

    if n_scenarios == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        scen_df = df[df["scenario"] == scen]
        sns.barplot(
            data=scen_df, x="success", y="Config", ax=ax,
            errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
        )
        for p in ax.patches:
            width = p.get_width()  # horizontal bar width
            ax.text(
                x=width + 0.01,  # slightly past the end of the bar
                y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
                s=f"{width * 100:.2f}%",  # percentage text
                ha='left',  # align left to avoid overlap
                va='center'  # vertically centered
            )
        ax.set_title(f"Scenario: {scen}", fontsize=13, pad=5)
        ax.set_xlabel("")
        ax.set_ylabel("Success Rate" if ax == axes[0] else "")
        ax.tick_params(axis="x")
        ax.margins(y=0.01)

    plt.tight_layout()
    plt.show()

    # --- 2. Detailed per-scenario plots ---
    metrics = {
        "Success Rate": "success",
        "Iteration Count": "num_RRT_iterations",
        "Trajectory Length": "trajectory_length",
        "Runtime": "runtime [sec]",
        # "Control Mean": "ctrl_effort_mean"
    }

    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False, constrained_layout=True)
        fig.suptitle(f"Scenario: {scen}", fontsize=15)

        for ax, (title, col) in zip(axes, metrics.items()):
            if title == "Success Rate":
                sns.barplot(
                    data=scen_df, x=col, y="Config", ax=ax,
                    errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
                )
                for p in ax.patches:
                    width = p.get_width()  # horizontal bar width
                    ax.text(
                        x=width + 0.01,  # slightly past the end of the bar
                        y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
                        s=f"{width * 100:.2f}%",  # percentage text
                        ha='left',  # align left to avoid overlap
                        va='center'  # vertically centered
                    )
            else:
                sns.boxplot(
                    data=scen_df, x=col, y="Config", ax=ax,
                    palette=PALETTE, hue='sampling_tech', legend=False, showfliers=False
                )
            ax.set_title(title, fontsize=13, pad=5)
            ax.set_xlabel("")
            ax.tick_params(axis="x")
            ax.margins(y=0.01)

        plt.tight_layout()
        plt.show()


def plot_results_success_only(df):
    # Filter for successful runs
    df_success = df[df["success"] == True].copy()

    # --- Build unified config labels ---
    df_success["Config"] = df_success.apply(
        lambda row: (
            f"MPPI({row['params']})"
            if row["algorithm"] == "MPPI"
            else f"DiTree-{row['sampling_tech']}({row['params']})"
        ),
        axis=1
    )

    sns.set(style="whitegrid", context="paper", font_scale=1.1)

    scenarios = df_success["scenario"].unique()
    metrics = {
        "Iteration Count": "num_RRT_iterations",
        "Trajectory Length": "trajectory_length",
        "Runtime": "runtime [sec]",
        # "Control Mean": "ctrl_effort_mean"
    }

    for scen in scenarios:
        scen_df = df_success[df_success["scenario"] == scen]

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), sharey=False, constrained_layout=True)
        fig.suptitle(f"Scenario (successful only): {scen}", fontsize=15)

        for ax, (title, col) in zip(axes, metrics.items()):
            sns.boxplot(
                data=scen_df, x=col, y="Config", ax=ax, palette=PALETTE,
                hue='sampling_tech', legend=False, showfliers=False
            )
            ax.set_title(title, fontsize=13, pad=5)
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), ha="center")

        plt.tight_layout()
        plt.show()


def plot_results_summarize_scenarios(df):
    # --- Build unified config labels ---
    df["Config"] = df.apply(
        lambda row: (
            f"MPPI({row['params']})"
            if row["algorithm"] == "MPPI"
            else f"DiTree-{row['sampling_tech']}({row['params']})"
        ),
        axis=1
    )
    df['scenario'] = 'Summarizing All Scenarios'
    scenarios = df["scenario"].unique()
    sns.set(style="whitegrid", context="paper", font_scale=1.1)

    # --- 1. Summary bar plots (success rate only) ---
    axes = plt.axes()

    sns.barplot(
        data=df, x="success", y="Config", ax=axes,
        errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
    )
    for p in axes.patches:
        width = p.get_width()  # horizontal bar width
        axes.text(
            x=width + 0.01,  # slightly past the end of the bar
            y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
            s=f"{width * 100:.2f}%",  # percentage text
            ha='left',  # align left to avoid overlap
            va='center'  # vertically centered
        )
    # Add percentages on top of each bar
    for p in axes.patches:
        width = p.get_width()  # horizontal bar width
        axes.text(
            x=width + 0.01,  # slightly past the end of the bar
            y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
            s=f"{width * 100:.2f}%",  # percentage text
            ha='left',  # align left to avoid overlap
            va='center'  # vertically centered
        )

    axes.set_title(f"Expectation over all scenarios", fontsize=13, pad=5)
    axes.set_xlabel("")
    axes.set_ylabel("Success Rate")
    axes.tick_params(axis="x")
    axes.margins(y=0.01)

    plt.tight_layout()
    plt.show()

    # --- 2. Detailed per-scenario plots ---
    metrics = {
        # "Success Rate": "success",
        "Iteration Count": "num_RRT_iterations",
        "Trajectory Length": "trajectory_length",
        "Runtime": "runtime [sec]",
        # "Control Mean": "ctrl_effort_mean"
    }

    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False, constrained_layout=True)
        fig.suptitle(f"Scenario: {scen}", fontsize=15)

        for ax, (title, col) in zip(axes, metrics.items()):
            if title == "Success Rate":
                sns.barplot(
                    data=scen_df, x=col, y="Config", ax=ax,
                    errorbar=None, palette=PALETTE, hue='sampling_tech', legend=False
                )
                for p in ax.patches:
                    width = p.get_width()  # horizontal bar width
                    ax.text(
                        x=width + 0.01,  # slightly past the end of the bar
                        y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
                        s=f"{width * 100:.2f}%",  # percentage text
                        ha='left',  # align left to avoid overlap
                        va='center'  # vertically centered
                    )
                for p in ax.patches:
                    width = p.get_width()  # horizontal bar width
                    ax.text(
                        x=width + 0.01,  # slightly past the end of the bar
                        y=p.get_y() + p.get_height() / 2,  # vertical center of the bar
                        s=f"{width * 100:.2f}%",  # percentage text
                        ha='left',  # align left to avoid overlap
                        va='center'  # vertically centered
                    )
            else:
                sns.boxplot(
                    data=scen_df, x=col, y="Config", ax=ax,
                    palette=PALETTE, hue='sampling_tech', legend=False, showfliers=False
                )
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
                                df = pd.concat([df, df_temp]) if df is not None else df_temp
    plot_results_summarize_scenarios(df.copy())
    plot_results_success_only(df.copy())
    plot_results(df.copy())
