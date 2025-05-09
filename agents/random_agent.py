import random
# Picks a random valid column, no strategy involved
def get_move(board, agent_disc=None, opponent_disc=None):
    valid_columns = [c for c in range(7) if board[0][c] == ' ']
    return random.choice(valid_columns) if valid_columns else None

