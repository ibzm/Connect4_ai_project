import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from game.board import ROWS, COLUMNS, is_valid_move, drop_disc, check_win
from agents.minimax_agent import get_move as minimax_move
import random

model = None
feature_cols = [f'cell_{i}' for i in range(42)]

def load_and_train_model():
    global model
    print("Loading model from midgame Minimax dataset...")
    df = pd.read_csv("data/minimax_dataset_v2.csv")
    X = df[feature_cols]
    y = df['move']

    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=300, random_state=42)
    model.fit(X, y)
    print(f"Trained on {len(X)} board states.")
    print(f"Dataset shape: {df.shape}")
    print(f"Move value counts:\n{y.value_counts().sort_index()}")


def board_to_features(board, agent_disc='●', opponent_disc='○'):
    flat = []
    for row in range(ROWS):
        for col in range(COLUMNS):
            cell = board[row, col]
            if cell == agent_disc:
                flat.append(1)
            elif cell == opponent_disc:
                flat.append(2)
            else:
                flat.append(0)
    return flat

def simulate_drop(board, col, disc):
    temp = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if temp[row, col] == ' ':
            temp[row, col] = disc
            return temp, row
    return board, None

def find_winning_move(board, disc):
    for col in range(COLUMNS):
        if is_valid_move(board, col):
            temp_board, row = simulate_drop(board, col, disc)
            if row is not None and check_win(temp_board, row, col, disc):
                return col
    return None

def get_move(board, agent_disc='●', opponent_disc='○'):
    global model
    if model is None:
        load_and_train_model()

    # Rule 1: Win immediately if possible
    winning_col = find_winning_move(board, agent_disc)
    if winning_col is not None:
        return winning_col

    # Rule 2: Block opponent's win
    block_col = find_winning_move(board, opponent_disc)
    if block_col is not None:
        return block_col

    # Rule 3: Use ML to select move
    valid_cols = [c for c in range(COLUMNS) if is_valid_move(board, c)]
    flat = board_to_features(board, agent_disc, opponent_disc)
    input_df = pd.DataFrame([flat], columns=feature_cols)

    try:
        predicted_move = model.predict(input_df)[0]
        if predicted_move in valid_cols:
            return predicted_move
    except Exception as e:
        print(f"Prediction error: {e}")

    # Fallback: choose first valid move
    if valid_cols:
        return valid_cols[0]

    # Rule 4: Fallback to Minimax (should rarely happen)
    return minimax_move(board, agent_disc, opponent_disc)

