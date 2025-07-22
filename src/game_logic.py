from src.cv import extract_board_image, classify_board_yolo, yolo_model

def check_win(board, player):
    win_pos = [(0,1,2), (3,4,5), (6,7,8),
               (0,3,6), (1,4,7), (2,5,8),
               (0,4,8), (2,4,6)]
    return any(all(board[i] == player for i in combo) for combo in win_pos)

def minimax(board, is_maximizing):
    if check_win(board, 'X'):
        return 1  # robot wins
    if check_win(board, 'O'):
        return -1  # player wins
    if ' ' not in board:
        return 0  # draw

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

def find_best_move(board):
    # Block O if they are about to win
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            if check_win(board, 'O'):
                board[i] = ' '
                return i  # block the win
            board[i] = ' '

    # Prioritize center
    if board.count(' ') == 9 or board[4] == ' ':
        return 4

    # Fallback to minimax
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



def get_board_state(image):
    board_img = extract_board_image(image)
    return classify_board_yolo(board_img, yolo_model)
