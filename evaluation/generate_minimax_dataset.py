import os
import sys
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add root project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import create_board, ROWS, COLUMNS, is_valid_move, drop_disc
from agents.minimax_agent import get_move as minimax_move

# Converts a board matrix into a flat list of numeric values
def board_to_flat(board, agent_disc='●', opponent_disc='○'):
    flat = []
    for row in range(ROWS):
        for col in range(COLUMNS):
            val = board[row, col]
            if val == agent_disc:
                flat.append(1)
            elif val == opponent_disc:
                flat.append(2)
            else:
                flat.append(0)
    return flat
# Creates a board with a random number of moves already played 
def generate_random_board(min_moves=15, max_moves=35):
    board = create_board()
    num_moves = random.randint(min_moves, max_moves)
    turn = 0

    for _ in range(num_moves):
        valid_cols = [c for c in range(COLUMNS) if is_valid_move(board, c)]
        if not valid_cols:
            break
        col = random.choice(valid_cols)
        disc = '●' if turn % 2 == 0 else '○'
        drop_disc(board, col, disc)
        turn += 1

    return board, turn

def simulate_game(_):
    board, turn = generate_random_board()
    agent_disc = '●' if turn % 2 == 0 else '○'
    opponent_disc = '○' if agent_disc == '●' else '●'

    move = minimax_move(board, agent_disc, opponent_disc)
    if move is None or not is_valid_move(board, move):
        return []  

    flat = board_to_flat(board, agent_disc, opponent_disc)
    return [flat + [move]]
 #Runs multiple simulations in parallel to generate a dataset of Minimax decisions
def generate_parallel_dataset(num_games=10000, output_path='data/minimax_dataset_v2.csv'):
    print(f"Generating {num_games} random Minimax board positions using {cpu_count() - 1} cores...\n")

    all_data = []
    seen_boards = set()  # To avoid duplicates

    with Pool(cpu_count() - 1) as pool:
        for i, game_data in enumerate(tqdm(pool.imap_unordered(simulate_game, range(num_games * 2)), total=num_games * 2)):
            if game_data:
                row = game_data[0]
                key = tuple(row[:-1])  

                if key not in seen_boards:
                    seen_boards.add(key)
                    all_data.append(row)
                # Stop when we've collected enough unique rows
                if len(all_data) >= num_games:
                    break

            if (i + 1) % 500 == 0:
                print(f"Collected {len(all_data)}/{num_games} unique boards")

    if not all_data:
        print("No valid board states were generated.")
        return

    df = pd.DataFrame(all_data)
    df.columns = [f'cell_{i}' for i in range(42)] + ['move']
    df.to_csv(output_path, index=False)
    print(f"\nSaved dataset with {len(df)} rows to {output_path}")

if __name__ == "__main__":
    generate_parallel_dataset(num_games=10000)
