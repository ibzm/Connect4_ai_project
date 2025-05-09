import math
import random
import numpy as np
from game.board import ROWS, COLUMNS, is_valid_move, drop_disc, check_win

# Keeps track of stats for minimax performance analysis
metrics = {
    'nodes_expanded': 0,
    'max_depth': 0,
    'branching_factors': []
}

def reset_metrics():
    metrics['nodes_expanded'] = 0
    metrics['max_depth'] = 0
    metrics['branching_factors'] = []
    
# Gets all columns where a move can still be made
def get_valid_columns(board):
    return [col for col in range(COLUMNS) if is_valid_move(board, col)]

# Checks if someone has already won
def check_winner(board):
    for row in range(ROWS):
        for col in range(COLUMNS):
            disc = board[row, col]
            if disc != ' ' and check_win(board, row, col, disc):
                return disc
    return None

# Checks if the game is over (win or full board)
def is_terminal_node(board):
    return check_winner(board) is not None or len(get_valid_columns(board)) == 0
# Assigns a score to the current board from the perspective of our agent
def score_position(board, agent_disc, opponent_disc):
    score = 0
    center_array = board[:, COLUMNS // 2]
    score += list(center_array).count(agent_disc) * 31
# Horizontal scoring
    for row in range(ROWS):
        for col in range(COLUMNS - 3):
            window = board[row, col:col + 4]
            score += evaluate_window(window, agent_disc, opponent_disc)
            
  # Vertical scoring
    for col in range(COLUMNS):
        for row in range(ROWS - 3):
            window = board[row:row + 4, col]
            score += evaluate_window(window, agent_disc, opponent_disc)
  # Diagonal scoring
    for row in range(ROWS - 3):
        for col in range(COLUMNS - 3):
            window = [board[row + i, col + i] for i in range(4)]
            score += evaluate_window(window, agent_disc, opponent_disc)
  # Diagonal scoring
    for row in range(3, ROWS):
        for col in range(COLUMNS - 3):
            window = [board[row - i, col + i] for i in range(4)]
            score += evaluate_window(window, agent_disc, opponent_disc)

    return score

def evaluate_window(window, agent_disc, opponent_disc):
    window = list(window)
    score = 0
    if window.count(agent_disc) == 4:
        return 100000
    elif window.count(agent_disc) == 3 and window.count(' ') == 1:
        score += 100
    elif window.count(agent_disc) == 2 and window.count(' ') == 2:
        score += 10

    if window.count(opponent_disc) == 4:
        return -100000 # blocking opponent's win
    elif window.count(opponent_disc) == 3 and window.count(' ') == 1:
        score -= 90
    elif window.count(opponent_disc) == 2 and window.count(' ') == 2:
        score -= 10

    return score
# Classic minimax algorithm with depth and evaluation
def minimax(board, depth, maximizingPlayer, agent_disc, opponent_disc, current_depth=0):
    metrics['nodes_expanded'] += 1
    metrics['max_depth'] = max(metrics['max_depth'], current_depth)

    valid_columns = get_valid_columns(board)
    metrics['branching_factors'].append(len(valid_columns))

    is_terminal = is_terminal_node(board)
    winner = check_winner(board)
  #game over or depth limit reached
    if depth == 0 or is_terminal:
        if winner == agent_disc:
            return (None, 1000000 - (4 - depth))
        elif winner == opponent_disc:
            return (None, -1000000 + (4 - depth))
        else:
            return (None, score_position(board, agent_disc, opponent_disc))
    # Maximize score for the agent
    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_columns)
        for col in valid_columns:
            temp_board = board.copy()
            drop_disc(temp_board, col, agent_disc)
            _, new_score = minimax(temp_board, depth - 1, False, agent_disc, opponent_disc, current_depth + 1)
            if new_score > value:
                value = new_score
                best_col = col
        return best_col, value
        # Minimize score for the opponent
    else:
        value = math.inf
        best_col = random.choice(valid_columns)
        for col in valid_columns:
            temp_board = board.copy()
            drop_disc(temp_board, col, opponent_disc)
            _, new_score = minimax(temp_board, depth - 1, True, agent_disc, opponent_disc, current_depth + 1)
            if new_score < value:
                value = new_score
                best_col = col
        return best_col, value

def get_move(board, agent_disc, opponent_disc):
    board_copy = board.copy()
    column, _ = minimax(board_copy, depth=4, maximizingPlayer=True, agent_disc=agent_disc, opponent_disc=opponent_disc)
    return column
