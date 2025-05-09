import os
import sys
import random
import time
import psutil
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Game setup
from game.board import create_board, is_valid_move, drop_disc, check_win, is_draw, ROWS, COLUMNS
from agents.random_agent import get_move as random_move
from agents.smart_agent import get_move as smart_move
from agents.minimax_agent import get_move as minimax_move, metrics as minimax_metrics, reset_metrics as reset_minimax_metrics
from agents.ml_agent import get_move as ml_move
from agents.ml_agent_2 import get_move as ml2_move

NUM_GAMES = 500 # Number of games to simulate
AGENTS = {
    "Random": random_move,
    "Smart": smart_move,
    "Minimax": minimax_move,
    "ML": ml_move,
    "ML2": ml2_move
}

# Plays one full game and records data like time, memory, win type, etc
def play_game(agent1_name, agent2_name):
    reset_minimax_metrics()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    board = create_board()
    turn = 0
    discs = ['●', '○']
    agents = [AGENTS[agent1_name], AGENTS[agent2_name]]
    agent_names = [agent1_name, agent2_name]

    total_nodes = 0
    max_depth = 0
    branching_factors = []

    timing_per_move = {agent1_name: [], agent2_name: []}
    scores = {discs[0]: 0, discs[1]: 0}
    mid_game_leader = None
    win_type = None

    while True:
        current = turn % 2
        move_func = agents[current]
        disc = discs[current]
        opp_disc = discs[1 - current]
        agent_name = agent_names[current]

        if agent_name == "Minimax":
            reset_minimax_metrics()

        board_input = board.copy()

        try:
            start_move = time.time()
            col = move_func(board_input, disc, opp_disc)
            move_time = time.time() - start_move
            current_mem = process.memory_info().rss
            move_mem_mb = (current_mem - mem_before) / (1024 ** 2)
            timing_per_move[agent_name].append((move_time, move_mem_mb))
        except Exception as e:
            print(f"Error during {agent_name}'s move: {e}")
            return {
                "winner": 1 - current, "length": turn, "reason": "crash",
                f"{agent1_name}_avg_time": 0,
                f"{agent2_name}_avg_time": 0,
                f"{agent1_name}_avg_mem": 0,
                f"{agent2_name}_avg_mem": 0
            }

        if col is None or not is_valid_move(board, col):
            return {
                "winner": 1 - current, "length": turn, "reason": "invalid",
                f"{agent1_name}_avg_time": 0,
                f"{agent2_name}_avg_time": 0,
                f"{agent1_name}_avg_mem": 0,
                f"{agent2_name}_avg_mem": 0
            }

        row, _ = drop_disc(board, col, disc)
        scores[disc] += 1

        result_type = check_win(board, row, col, disc)
        if result_type:
            win_type = result_type
            break
        elif is_draw(board):
            break

 # See who’s ahead at the halfway point
        if turn + 1 == (ROWS * COLUMNS) // 2:
            if scores[discs[0]] > scores[discs[1]]:
                mid_game_leader = discs[0]
            elif scores[discs[1]] > scores[discs[0]]:
                mid_game_leader = discs[1]
            else:
                mid_game_leader = "draw"

        if agent_name == "Minimax":
            total_nodes += minimax_metrics['nodes_expanded']
            max_depth = max(max_depth, minimax_metrics['max_depth'])
            branching_factors.extend(minimax_metrics['branching_factors'])

        turn += 1

    avg_time_1 = np.mean([t for t, _ in timing_per_move[agent1_name]])
    avg_mem_1 = np.mean([m for _, m in timing_per_move[agent1_name]])
    avg_time_2 = np.mean([t for t, _ in timing_per_move[agent2_name]])
    avg_mem_2 = np.mean([m for _, m in timing_per_move[agent2_name]])

    result = {
        "winner": current,
        "length": turn + 1,
        "reason": "win",
        "win_type": win_type,
        "mid_leader": mid_game_leader,
        f"{agent1_name}_avg_time": avg_time_1,
        f"{agent2_name}_avg_time": avg_time_2,
        f"{agent1_name}_avg_mem": avg_mem_1,
        f"{agent2_name}_avg_mem": avg_mem_2
    }

    if "Minimax" in [agent1_name, agent2_name]:
        result.update({
            "nodes": total_nodes,
            "depth": max_depth,
            "branching": sum(branching_factors) / max(len(branching_factors), 1)
        })

    return result

# Runs multiple games for a pair of agents
def run_matchup(agent1_name, agent2_name, games=NUM_GAMES):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(play_game, [(agent1_name, agent2_name)] * games)

    df = pd.DataFrame(results)
    df["Agent 1"] = agent1_name
    df["Agent 2"] = agent2_name
    return df

# Generates all evaluation graphs from the data
def plot_graphs(df, agent1_name, agent2_name, output_dir="evaluation_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    df = df[(df["Agent 1"] == agent1_name) & (df["Agent 2"] == agent2_name)]
    if df.empty:
        print(f"No data to plot for {agent1_name} vs {agent2_name}.")
        return
  # Win/Draw Summary
    summary = df.groupby(["Agent 1", "Agent 2", "winner"]).size().unstack(fill_value=0)
    summary["Draw Rate"] = summary.get(-1, 0) / summary.sum(axis=1)
    summary["Agent 1 Win Rate"] = summary.get(0, 0) / summary.sum(axis=1)
    summary["Agent 2 Win Rate"] = summary.get(1, 0) / summary.sum(axis=1)
    summary.index = [f"{a1} vs {a2}" for a1, a2 in summary.index]
    summary[["Agent 1 Win Rate", "Agent 2 Win Rate", "Draw Rate"]].plot(kind="bar", figsize=(10, 6), title="How Often Each Agent Wins or Draws")
    plt.ylabel("Proportion")
    plt.xlabel("Matchup")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_metrics.png"))
    plt.close()

    if agent1_name == "Minimax" or agent2_name == "Minimax":
        df_minimax = df[(df["Agent 1"] == "Minimax") | (df["Agent 2"] == "Minimax")]
        if not df_minimax.empty:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(df_minimax["nodes"], bins=20, ax=axes[0])
            axes[0].set_title("Total Nodes Explored by Minimax")
            sns.histplot(df_minimax["depth"], bins=10, ax=axes[1])
            axes[1].set_title("Deepest Search Level Reached")
            sns.histplot(df_minimax["branching"], bins=10, ax=axes[2])
            axes[2].set_title("Average Branching per Minimax Move")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "search_performance_metrics.png"))
            plt.close()
 # Efficiency metrics: time and memory
    time_cols = [col for col in df.columns if col.endswith("_avg_time")]
    time_df = df[time_cols].melt(var_name="Agent", value_name="Avg Move Time")
    time_df["Agent"] = time_df["Agent"].str.replace("_avg_time", "")

    mem_cols = [col for col in df.columns if col.endswith("_avg_mem")]
    mem_df = df[mem_cols].melt(var_name="Agent", value_name="Avg Mem (MB)")
    mem_df["Agent"] = mem_df["Agent"].str.replace("_avg_mem", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=time_df, x="Agent", y="Avg Move Time", ax=axes[0])
    axes[0].set_title("Average Thinking Time per Move")
    sns.boxplot(data=mem_df, x="Agent", y="Avg Mem (MB)", ax=axes[1])
    axes[1].set_title("Average Memory Used per Move")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_metrics.png"))
    plt.close()
# Type of wins (horizontal, vertical, diagonal)
    if "win_type" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="win_type")
        plt.title("What Kind of Wins Happened?")
        plt.xlabel("Win Pattern")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "win_type_distribution.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="length", bins=20)
    plt.title("How Long Did the Games Last?")
    plt.xlabel("Number of Moves")
    plt.ylabel("Game Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "game_length_distribution.png"))
    plt.close()

    if "mid_leader" in df.columns and "winner" in df.columns:
        disc_map = {'●': 0, '○': 1}
        df['Mid Equals Winner'] = df.apply(
            lambda row: disc_map.get(row['mid_leader']) == row['winner'] if row['mid_leader'] in disc_map else None,
            axis=1
        )

    if "Mid Equals Winner" in df.columns:
        filtered_df = df[df["Mid Equals Winner"].isin([True, False])]
    if not filtered_df.empty:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=filtered_df, x="Mid Equals Winner")
        plt.title("Was the Mid-Game Leader Also the Final Winner?")
        plt.xlabel("Mid Leader == Final Winner")
        plt.ylabel("Game Count")
        plt.xticks([0, 1], ["No", "Yes"])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mid_leader_win_consistency.png"))
        plt.close()


def get_matchups_from_input():
    parser = argparse.ArgumentParser(description="Evaluate Connect 4 agents.")
    parser.add_argument("agent1", nargs="?", help="Name of Agent 1", choices=AGENTS.keys())
    parser.add_argument("agent2", nargs="?", help="Name of Agent 2", choices=AGENTS.keys())
    args = parser.parse_args()

    if args.agent1 and args.agent2:
        return [(args.agent1, args.agent2)]

    print("Available agents:", ", ".join(AGENTS.keys()))
    a1 = input("Enter Agent 1 name: ").strip()
    a2 = input("Enter Agent 2 name: ").strip()
    if a1 not in AGENTS or a2 not in AGENTS:
        print(f"Invalid agent(s). Please choose from: {', '.join(AGENTS.keys())}")
        sys.exit(1)
    return [(a1, a2)]

def main():
    all_results = []
    matchups = get_matchups_from_input()

    for a1, a2 in matchups:
        print(f"Simulating {a1} vs {a2}...")
        df = run_matchup(a1, a2)
        all_results.append((df, a1, a2))

    for df, a1, a2 in all_results:
        df.to_csv("evaluation_results_detailed.csv", index=False)
        print("\nSaved: evaluation_results_detailed.csv")
        plot_graphs(df, a1, a2)

    print("Graphs saved in evaluation_outputs/")

if __name__ == "__main__":
    main()
