import random
from src.cv import extract_board_image, classify_board_yolo, yolo_model


# --- Board Utilities ---
def check_win(board, player):
    win_pos = [(0,1,2), (3,4,5), (6,7,8),
               (0,3,6), (1,4,7), (2,5,8),
               (0,4,8), (2,4,6)]
    return any(all(board[i] == player for i in combo) for combo in win_pos)


# --- Minimax Algorithm for Hard Difficulty ---
def minimax(board, is_maximizing):
    if check_win(board, 'X'):
        return 1
    if check_win(board, 'O'):
        return -1
    if ' ' not in board:
        return 0

    if is_maximizing:
        best = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(board, False)
                board[i] = ' '
                best = max(best, score)
        return best
    else:
        best = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(board, True)
                board[i] = ' '
                best = min(best, score)
        return best


# --- Move Strategies ---
def random_move(board):
    return random.choice([i for i, cell in enumerate(board) if cell == ' '])


def medium_move(board):
    # Block O if about to win
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            if check_win(board, 'O'):
                board[i] = ' '
                return i
            board[i] = ' '

    # Prioritize center
    if board.count(' ') == 9 or board[4] == ' ':
        return 4

    # Prefer corners
    for i in [0, 2, 6, 8]:
        if board[i] == ' ':
            return i

    # Fallback: pick first free cell
    for i in range(9):
        if board[i] == ' ':
            return i


def hard_move(board):
    best_score = -float('inf')
    best_move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'X'
            score = minimax(board, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                best_move = i
    return best_move


# --- Move Selector ---
def select_move(board, difficulty):
    if difficulty == 1:
        return random_move(board)
    elif difficulty == 2:
        return medium_move(board)
    elif difficulty == 3:
        return hard_move(board)
    else:
        raise ValueError("Difficulty must be 1 (Easy), 2 (Medium), or 3 (Hard)")


# --- Board State Extraction ---
def get_board_state(image):
    board_img = extract_board_image(image)
    return classify_board_yolo(board_img, yolo_model)
