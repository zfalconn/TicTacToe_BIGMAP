import cv2
from src.cv import capture_frame, extract_board_image, classify_board_yolo, yolo_model
from src.game_logic import find_best_move
from src.bigmap_robot_opcua.robot.Yaskawa_YRC1000_OPCUA_client import Yaskawa_YRC1000
import time

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
    "home_to_play" : 'TICTACTOE_X0_HOME_PLAY',
    "play_to_home" : 'TICTACTOE_X0_PLAY_HOME'
}

def loop_live():
    
    try: 
        ### ROBOT INIT ###
        robot_url = "opc.tcp://192.168.1.20:16448"
        robot = Yaskawa_YRC1000(robot_url)
        print(" >> Robot initialized <<")
        print(robot.get_available_jobs())
        print(robot.set_servo(True))
        ### END INIT ###

        # WAIT FOR GO AHEAD FROM USER
        while True:
            user_start = input("Ready to play?: (Y/N)")
            if user_start.lower() == "y":
                print(robot.start_job(other_job["home_to_play"], True)) #MAKE ROBOT GO TO PLAY POSITION
                break
            elif user_start.lower() == "n":
                print("Program terminated by user.")
                break
            else:
                print("Please try again with valid input: Y for yes and N for no.")
        
        ### CAM INIT ###
        print("🔄 Starting webcam Tic Tac Toe loop...")
        print("➡️  Press 'c' to capture and analyze the board")
        print("❌ Press 'q' to quit")
        cap = cv2.VideoCapture(0)
        ### END INIT ###

        ### CV CONTROL LOOP ###
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
                robot.start_job(other_job["play_to_home"], True)
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
                        ### PERFORM MOVEMENT ###
                        print(robot.start_job(index_to_robot_job[best_move], block=True))
                        
                        print(" >> ROBOT MOVED! << ")



                    print("\n➡️  Press 'c' again to capture a new board, or 'q' to quit.")
                    time.sleep(0.5)

                except Exception as e:
                    print("🚨 Error:", str(e))

        ### END LOOP ###

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"ERROR: {e}")
    except user_start == "n":
        print(robot.set_servo(False))
        robot.stop_communication()
        print("Program ended. Robot disconnected.")
        # pass
    finally:
        print(robot.set_servo(False))
        robot.stop_communication()
        print("Program ended. Robot disconnected.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    loop_live()
