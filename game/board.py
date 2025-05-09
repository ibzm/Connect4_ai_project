import numpy as np
# Define the board dimensions
ROWS = 6
COLUMNS = 7
# Create an empty game board filled with spaces
def create_board():
    return np.full((ROWS, COLUMNS), ' ')

def print_board(board):
    for row in board:
        print('| ' + ' | '.join(row) + ' |')
    print('  ' + '   '.join(map(str, range(COLUMNS))))
# Check if a move in the given column is allowed 
def is_valid_move(board, col):
    return board[0, col] == ' '


def drop_disc(board, col, disc):
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == ' ':
            board[row, col] = disc
            return row, col
    return None, None
# Drop a disc into the selected column

# Check if the last move made results in a win
def check_win(board, row, col, disc):
    directions = [
        ((0, 1), "horizontal"),
        ((1, 0), "vertical"),
        ((1, 1), "diagonal"),
        ((1, -1), "diagonal")
    ]
    for (dr, dc), label in directions:
        count = 1
        for i in range(1, 4):
            r, c = row + dr*i, col + dc*i
            if 0 <= r < ROWS and 0 <= c < COLUMNS and board[r, c] == disc:
                count += 1
            else:
                break
        for i in range(1, 4):
            r, c = row - dr*i, col - dc*i
            if 0 <= r < ROWS and 0 <= c < COLUMNS and board[r, c] == disc:
                count += 1
            else:
                break
        if count >= 4:
            return label
    return None

# Check if the board is completely full with no winner 
def is_draw(board):
    return np.all(board[0] != ' ')
