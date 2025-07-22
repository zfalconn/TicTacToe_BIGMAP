from ultralytics import YOLO

model = YOLO("yolo11s.pt")

if __name__ == '__main__':
    model.train(data=r"..\data\TicTacToe.v3-tictactoe-v2-15-gray.yolov11\data.yaml", 
                epochs=500, patience=100, batch=16, 
                device=0, optimizer='AdamW', seed=42, 
                cos_lr=True, 
                project=r"..\models")

