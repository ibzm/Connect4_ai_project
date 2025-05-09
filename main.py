from game.board import ROWS, COLUMNS, create_board, print_board, is_valid_move, drop_disc, check_win, is_draw
from agents.random_agent import get_move as random_move
from agents.smart_agent import get_move as smart_move
from agents.minimax_agent import get_move as minimax_move
from agents.ml_agent import get_move as ml_move
from agents.ml_agent_2 import get_move as ml2_move

#give player list of agents
def choose_player(player_num):
    print(f"\nChoose Player {player_num}:")
    print("1. Human")
    print("2. Random Agent")
    print("3. Smart Agent")
    print("4. Minimax Agent")
    print("5. ML Agent")
    print("6. ML Agent (Minimax-Trained)")

    
    while True:
        try:
            choice = int(input("Enter your choice (1-6): "))
            if choice in [1, 2, 3, 4, 5, 6]:
                return choice
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input. Enter a number.")
#decide column for next move based on player type 
def get_column(player_type, board, disc, opponent_disc):
    if player_type == 1:
        try:
            return int(input("Choose a column (0–6): "))
        except ValueError:
            return -1
    elif player_type == 2:
        return random_move(board)
    elif player_type == 3:
        return smart_move(board, disc, opponent_disc)
    elif player_type == 4:
        return minimax_move(board, disc, opponent_disc)
    elif player_type == 5:
        return ml_move(board, disc, opponent_disc)
    elif player_type == 6:
        return ml2_move(board, disc, opponent_disc)


    
def main():
    player1_type = choose_player(1)
    player2_type = choose_player(2)

    player_type_names = {
        1: "Human",
        2: "Random Agent",
        3: "Smart Agent",
        4: "Minimax Agent",
        5: "ML Agent",
        6: "ML Agent 2 (Minimax-Trained)"

    }

    player1_disc = '●'
    player2_disc = '○'

    player_info = {
        0: {
            "name": f"Player 1 ({player_type_names[player1_type]} - {player1_disc})",
            "type": player1_type,
            "disc": player1_disc
        },
        1: {
            "name": f"Player 2 ({player_type_names[player2_type]} - {player2_disc})",
            "type": player2_type,
            "disc": player2_disc
        }
    }


    print("\n Player Setup:")
    print(player_info[0]["name"])
    print(player_info[1]["name"])

    
    board = create_board()
    print_board(board)
    game_over = False
    turn = 0  #keeps track of whose turn it is

    while not game_over:
        player_index = turn % 2
        current_player = player_info[player_index]
        opponent_player = player_info[1 - player_index]

        print(f"\n {current_player['name']}'s turn.")
        #validate move 
        col = get_column(current_player["type"], board, current_player["disc"], opponent_player["disc"])

        if col is None or col < 0 or col >= COLUMNS:
            print("Invalid move.")
            continue

        if not is_valid_move(board, col):
            print("Column full. Try again.")
            continue

        row, col = drop_disc(board, col, current_player["disc"])
        print_board(board)

        if check_win(board, row, col, current_player["disc"]):
            print(f"\n {current_player['name']} wins!")
            game_over = True
        elif is_draw(board):
            print("\n It's a draw!")
            game_over = True
        else:
            turn += 1

if __name__ == "__main__":
    main()
