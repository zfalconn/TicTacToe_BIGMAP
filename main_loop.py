import cv2
from src.cv import capture_frame, extract_board_image, classify_board_yolo, yolo_model
from src.game_logic import find_best_move
import time

index_to_position = {
    0: 1, 1: 2, 2: 3,
    3: 4, 4: 5, 5: 6,
    6: 7, 7: 8, 8: 9
}

def loop_live():
    print("🔄 Starting webcam Tic Tac Toe loop...")
    print("➡️  Press 'c' to capture and analyze the board")
    print("❌ Press 'q' to quit")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam")
            break

        # Show live preview
        display = frame.copy()
        cv2.putText(display, "Press 'c' to analyze, 'q' to quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Live Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Exiting...")
            break
        elif key == ord('c'):
            print("📸 Capturing frame and analyzing...")
            try:
                # Extract and classify board from current frame
                board_img = extract_board_image(frame)
                board = classify_board_yolo(board_img, yolo_model)

                print("\n📋 Detected Board:")
                for i in range(0, 9, 3):
                    print(board[i:i+3])

                best_move = find_best_move(board)
                if best_move == -1:
                    print("✅ Game over or no valid moves.")
                else:
                    print(f"🔷 Robot should draw 'X' at position: {index_to_position[best_move]}")

                print("\n➡️  Press 'c' again to capture a new board, or 'q' to quit.")
                time.sleep(0.5)

            except Exception as e:
                print("🚨 Error:", str(e))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    loop_live()
