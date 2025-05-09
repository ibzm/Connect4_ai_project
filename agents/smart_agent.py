import random
from game.board import ROWS, COLUMNS, is_valid_move, drop_disc, check_win

# Smart agent: tries to win, blocks opponent, otherwise plays randomly
def get_move(board, agent_disc, opponent_disc):
      # First, check if we can win in one move
    for col in range(COLUMNS):
        if is_valid_move(board, col):
            temp_board = board.copy()
            row, _ = drop_disc(temp_board, col, agent_disc)
            if check_win(temp_board, row, col, agent_disc):
                return col
    # If not, check if we need to block the opponent from winning
    for col in range(COLUMNS):
        if is_valid_move(board, col):
            temp_board = board.copy()
            row, _ = drop_disc(temp_board, col, opponent_disc)
            if check_win(temp_board, row, col, opponent_disc):
                return col

    # Otherwise, just pick a random valid column
    valid_columns = [c for c in range(COLUMNS) if is_valid_move(board, c)]
    return random.choice(valid_columns) if valid_columns else None

