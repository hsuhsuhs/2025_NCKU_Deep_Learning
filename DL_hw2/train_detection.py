from ultralytics import YOLO
import torch

def train_ksdd2_detection():
    # 確認 CUDA 是否可用
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 載入預訓練模型
    model = YOLO('yolov8n.pt') 

    # 開始訓練，imgsz 設為 640
    model.train(
        data='ksd_data.yaml',
        epochs=100,           # 工業缺陷需要較多 epoch
        imgsz=640,
        batch=16,
        device=device,
        project='runs/detect',
        name='ksdd2_yolo_train'
    )

    # 驗證並印出 mAP@0.5 
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50}")

if __name__ == '__main__':
    train_ksdd2_detection()