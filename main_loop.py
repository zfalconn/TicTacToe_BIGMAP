import cv2
import time
import sys
from src.cv import extract_board_image, classify_board_yolo, yolo_model
from src.game_logic import select_move, check_win
from src.bigmap_robot_opcua.robot.Yaskawa_YRC1000_OPCUA_client import Yaskawa_YRC1000


index_to_position = {
    0: 1, 1: 2, 2: 3,
    3: 4, 4: 5, 5: 6,
    6: 7, 7: 8, 8: 9
}

index_to_robot_job = {
    0: 'TICTACTOE_X1', 1: 'TICTACTOE_X2', 2: 'TICTACTOE_X3',
    3: 'TICTACTOE_X4', 4: 'TICTACTOE_X5', 5: 'TICTACTOE_X6',
    6: 'TICTACTOE_X7', 7: 'TICTACTOE_X8', 8: 'TICTACTOE_X9'
}

other_job = {
    "home_to_play": 'TICTACTOE_X0_HOME_PLAY',
    "play_to_home": 'TICTACTOE_X0_PLAY_HOME',
    "robot_win": 'TICTACTOE_WIN',
    "robot_lose": 'TICTACTOE_LOOSE'
}


def wait_for_user_to_start(robot, job_name: str) -> bool:
    while True:
        user_start = input("Ready to play? (Y/N): ").strip().lower()
        if user_start == "y":
            print(robot.start_job(job_name, True))
            return True
        elif user_start == "n":
            print(robot.set_servo(False))
            robot.stop_communication()
            print("Program terminated by user.")
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")


def get_user_difficulty():
    while True:
        try:
            difficulty = int(input("Select difficulty level (1: Easy, 2: Medium, 3: Hard): "))
            if difficulty in [1, 2, 3]:
                return difficulty
            else:
                print("Invalid input. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def display_board(board):
    print("\n📋 Current Board:")
    for i in range(0, 9, 3):
        print(board[i:i + 3])


def play_round(robot):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Could not open camera. Exiting.")
        return

    difficulty_level = get_user_difficulty()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam")
            break

        display = frame.copy()
        cv2.putText(display, "Press 'c' to capture, 'q' to quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Live Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Exiting game loop...")
            robot.start_job(other_job["play_to_home"], True)
            break

        elif key == ord('c'):
            try:
                print("📸 Capturing and analyzing board...")
                board_img = extract_board_image(frame)
                board = classify_board_yolo(board_img, yolo_model)
                display_board(board)

                if check_win(board, 'O'):
                    print("🎉 Player wins!")
                    robot.start_job(other_job["robot_lose"], True)
                    break
                if check_win(board, 'X'):
                        print("🏆 Robot wins!")
                        robot.start_job(other_job["robot_win"], True)
                        break
                elif ' ' not in board:
                    print("🤝 It's a draw!")
                    break

                best_move = select_move(board, difficulty_level)
                if best_move == -1:
                    print("✅ No valid moves left.")
                    break

                print(f"🔷 Robot should draw 'X' at position: {index_to_position[best_move]}")
                print(robot.start_job(index_to_robot_job[best_move], block=True))
                print("🦾 Robot moved.")

                time.sleep(0.5)

                # Re-capture board to check win condition for robot
                ret2, frame2 = cap.read()
                if ret2:
                    board_img2 = extract_board_image(frame2)
                    new_board = classify_board_yolo(board_img2, yolo_model)
                    display_board(new_board)

                    if check_win(new_board, 'X'):
                        print("🏆 Robot wins!")
                        robot.start_job(other_job["robot_win"], True)
                        break
                    elif ' ' not in new_board:
                        print("🤝 It's a draw!")
                        break

                print("➡️ Press 'c' to continue, or 'q' to quit.")

            except Exception as e:
                print("🚨 Error during board capture or move:", str(e))

    cap.release()
    cv2.destroyAllWindows()


def loop_live():
    try:
        robot_url = "opc.tcp://192.168.1.20:16448"
        robot = Yaskawa_YRC1000(robot_url)
        print("🤖 Robot initialized.")
        print(robot.get_available_jobs())
        print(robot.set_servo(True))

        if not wait_for_user_to_start(robot, other_job["home_to_play"]):
            sys.exit()

        while True:
            play_round(robot)

            again = input("Play again? (Y/N): ").strip().lower()
            if again != 'y':
                break
            print("Restarting game...")

        print(robot.start_job(other_job["play_to_home"], True))

    except Exception as e:
        print(f"🔥 ERROR: {e}")

    finally:
        print(robot.set_servo(False))
        robot.stop_communication()
        print("Robot disconnected. Program terminated.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    loop_live()
