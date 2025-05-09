import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from game.board import ROWS, COLUMNS, is_valid_move
import os

model = None

def load_and_train_model():
    global model
    print(" Training ML model from UCI Connect-4 dataset...")

    # Load UCI dataset
    col_names = [f'cell_{i}' for i in range(42)] + ['outcome']
    df = pd.read_csv('data/connect-4.data', names=col_names)

 
    df = df.replace({'x': 1, 'o': 2, 'b': 0}).infer_objects(copy=False)

    X = []
    y = []

    for _, row in df.iterrows():
        board_flat = row[:-1].tolist()  
        for col in range(COLUMNS):
            for row_i in reversed(range(ROWS)):
                index = row_i * COLUMNS + col
                if board_flat[index] == 1:  
                    X.append(board_flat)
                    y.append(col)  
                    break
            else:
                continue
            break

    if len(X) == 0:
        raise RuntimeError(" No training data extracted from dataset.")

    print(f" Extracted {len(X)} training samples.")

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X, y)
    model = clf
    print(" Model trained to predict best next move.")

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

def get_move(board, agent_disc='●', opponent_disc='○'):
    global model
    if model is None:
        load_and_train_model()

    features = board_to_features(board, agent_disc, opponent_disc)
    predicted_col = model.predict([features])[0]

    if is_valid_move(board, predicted_col):
        return predicted_col
    else:
        #choose first valid column if prediction is blocked
        for col in range(COLUMNS):
            if is_valid_move(board, col):
                return col
        return None
