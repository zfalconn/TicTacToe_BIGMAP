# main.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import cv2
from src.game_logic import get_board_state, find_best_move

# Board index to position mapping (1–9 layout)
index_to_position = {
    0: 1, 1: 2, 2: 3,
    3: 4, 4: 5, 5: 6,
    6: 7, 7: 8, 8: 9
}

def test_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"❌ Failed to load image: {path}")
        return

    try:
        #cells = extract_board_cells(image)
        board = get_board_state(image)  # now uses YOLO

        print("\n📋 Detected Board:")
        for i in range(0, 9, 3):
            print(board[i:i+3])

        best_move = find_best_move(board)
        if best_move == -1:
            print("✅ Game over or no valid moves.")
        else:
            position = index_to_position[best_move]
            print(f"\n🔷 Robot should draw 'X' at position: {position}")

    except Exception as e:
        print("🚨 Error:", str(e))

if __name__ == "__main__":
    test_image("test_imgs/6.jpg")  # Change path as needed
