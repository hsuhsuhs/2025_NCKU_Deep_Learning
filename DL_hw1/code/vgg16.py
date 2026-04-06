import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from multiprocessing import freeze_support # 導入 freeze_support
import copy # 用於複製最佳模型

# PyTorch 相關導入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, models, transforms
from torchvision.models import VGG16_Weights

# Sklearn 相關導入
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold # 導入 KFold

# ==============================================================================
#  全域常數與路徑 

# --- 路徑設定 ---
ORIGINAL_DATA_DIR = 'Simple_chinastell_data' 
BASE_WORKING_DIR = '.'
DATA_DIR = os.path.join(BASE_WORKING_DIR, 'data_split_pytorch')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = os.path.join(BASE_WORKING_DIR, 'output_pytorch_kfold') # 輸出資料夾

# --- 超參數 ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50 # 每個 Fold 的最大 Epochs
TEST_SPLIT_RATIO = 0.2
K_FOLDS = 4 # K-Fold 的折數
PATIENCE = 10 # EarlyStopping 的耐心值

# --- 獲取類別名稱 ---
if not os.path.exists(ORIGINAL_DATA_DIR):
    print(f"警告：找不到原始資料路徑 {ORIGINAL_DATA_DIR}。")
    CLASS_NAMES = []
    NUM_CLASSES = 0
else:
    CLASS_NAMES = sorted([d for d in os.listdir(ORIGINAL_DATA_DIR) if os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, d))])
    NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================================
# 模型定義
 
class VGG16(nn.Module):
    def __init__(self, num_classes=10): 
        super(VGG16, self).__init__()
        
        # 這部分將被凍結，並載入預訓練權重
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5: 14 -> 7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 'classifier'這部分將被訓練
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # 512 * 7 * 7 = 25088
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    # 'forward' 定義資料如何流過模型
    def forward(self, x):
        x = self.features(x) # 讓資料流過特徵層
        x = torch.flatten(x, 1) # 將 7x7x512 的特徵圖攤平成一維向量
        x = self.classifier(x) # 讓攤平的向量流過 'classifier' 
        return x

# ==============================================================================
#  輔助函式定義 

# 將原始資料夾的圖片按照 8:2 比例分割成訓練集和測試集。
def split_data(original_dir, train_dir, test_dir, split_ratio):
    
    if (os.path.exists(train_dir) and any(os.scandir(train_dir))) or (os.path.exists(test_dir) and any(os.scandir(test_dir))):
        print("資料夾 'train'/'test' 已有內容，跳過資料分割步驟。")
        return

    print(f"正在分割資料，比例 (Train/Test): {1-split_ratio}/{split_ratio}")
    for class_name in CLASS_NAMES:
        original_class_path = os.path.join(original_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)

        # 建立目標資料夾
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # 取得所有圖片檔案
        all_files = [f for f in os.listdir(original_class_path) if os.path.isfile(os.path.join(original_class_path, f))]
        random.shuffle(all_files)

        # 計算 80% 的分割點
        split_index = int(len(all_files) * (1 - split_ratio))
        train_files = all_files[:split_index] # 前 80%
        test_files = all_files[split_index:]  # 後 20%

        # 將檔案複製到新資料夾
        for f in train_files:
            shutil.copy(os.path.join(original_class_path, f), os.path.join(train_class_path, f))
        for f in test_files:
            shutil.copy(os.path.join(original_class_path, f), os.path.join(test_class_path, f))
                        
    print("資料分割完成。")

# 繪製 K-Fold 的平均訓練曲線  (這裡簡化只繪製最後一折的訓練曲線)
def plot_training_history(history_list, output_dir):

    if not history_list:
        return
        
    history = history_list[-1] # 只拿最後一折的 history 來畫圖
    acc = [h for h in history['train_acc']]
    val_acc = [h for h in history['val_acc']]
    loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Last Fold (Fold {len(history_list)}) Training History (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Last Fold (Fold {len(history_list)}) Training History (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.suptitle('Model Training History (PyTorch K-Fold)')
    save_path = os.path.join(output_dir, 'training_history_curves_kfold_last.png')
    plt.savefig(save_path)
    print(f"最後一折的訓練曲線圖已儲存至: {save_path}")


# ==============================================================================
#  載入權重到自訂的 VGG16 class中
def create_model(device):
    
    #  取得權重
    official_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    #  建立自訂的 VGG16 骨架 (出口為 10)
    custom_model = VGG16(num_classes=NUM_CLASSES)
    #  複製 'features' - 權重完美匹配
    custom_model.features.load_state_dict(official_model.features.state_dict())
    #  複製 'classifier' 的共同層
    custom_model.classifier[0].load_state_dict(official_model.classifier[0].state_dict())
    custom_model.classifier[3].load_state_dict(official_model.classifier[3].state_dict())
    #  凍結特徵層
    for param in custom_model.features.parameters():
        param.requires_grad = False
        
    # 將的模型送上 GPU/CPU
    custom_model = custom_model.to(device)
    
    print("自訂 VGG16 模型建立並載入完畢。")
    return custom_model

# ==============================================================================
#  主執行函式 
def main_process():
    """ 
    所有主要的執行邏輯都在這裡。
    """
    
    # --- 0. 啟動時的輸出 ---
    print(f"PyTorch Version: {torch.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current GPU Name: {torch.cuda.get_device_name(0)}")

    if NUM_CLASSES == 0:
        print(f"錯誤：找不到原始資料路徑 {ORIGINAL_DATA_DIR} 或資料夾為空。")
        exit()
    else:
        print(f"成功找到 {NUM_CLASSES} 個類別: {CLASS_NAMES}")

    # 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 階段一：資料準備 ---
    print("\n--- 階段一：資料準備 (8:2 分割) ---")
    
    # 1. 執行 8:2 資料分割
    split_data(ORIGINAL_DATA_DIR, TRAIN_DIR, TEST_DIR, TEST_SPLIT_RATIO)

    # 2. 影像轉換
    data_transforms = {
        # 訓練集
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomRotation(30), # 隨機旋轉 30 度
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # 隨機裁切與縮放
            transforms.RandomHorizontalFlip(), # 隨機水平翻轉
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 標準化
        ]),

        #  驗證集/ 測試集
        'val': transforms.Compose([ # 驗證集/測試集不需增強
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE), # 僅中心裁切以確保評估的一致性
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 標準化
        ]),
    }

    # 3. 載入 80% TRAIN_DIR 和 20% TEST_DIR
    # K-Fold 只會對 `train_full_dataset` 操作
    train_full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    # 測試集 (Test Set)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms['val']) 
    
    # 為了 K-Fold 驗證，我們需要一個 '乾淨' (無增強) 的版本
    val_clean_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['val'])

    print(f"已載入 {len(train_full_dataset)} 張圖片 (80%) 用於 K-Fold 訓練/驗證。")
    print(f"已載入 {len(test_dataset)} 張圖片 (20%) 用於最終測試 (已鎖定)。")

    # --- 階段二：K-Fold 交叉驗證 ---
    print(f"\n--- 階段二：開始 {K_FOLDS}-Fold 交叉驗證 ---")
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_fold_histories = []
    
    # K-Fold 迴圈 (Outer Loop)
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_full_dataset)):
        print(f"\n==================== FOLD {fold + 1}/{K_FOLDS} ====================")
        
        # 建立此 Fold 的資料子集 
        # 訓練集使用 'train' 轉換 (有增強)
        train_subset = Subset(train_full_dataset, train_indices)
        # 驗證集使用 'val' 轉換 (無增強)
        val_subset = Subset(val_clean_dataset, val_indices) 
        
        # 建立資料載入器
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=2, pin_memory=True, persistent_workers=True)
                                  
        print(f"此 Fold 訓練集: {len(train_subset)} 張圖片")
        print(f"此 Fold 驗證集: {len(val_subset)} 張圖片")

        # 建立一個全新的、載入好權重的模型
        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        
        # 優化器只訓練 model.classifier 的參數 
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

        # 訓練此 Fold
        best_val_loss = float('inf')
        epochs_no_improve = 0
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_epoch_acc = 0.0

        # Epoch 迴圈 (Inner Loop)
        for epoch in range(EPOCHS):
            model.train() 
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(train_subset) 
            epoch_acc = running_corrects.double() / len(train_subset)
            fold_history['train_loss'].append(epoch_loss)
            fold_history['train_acc'].append(epoch_acc.item())

            model.eval() 
            val_running_loss = 0.0
            val_running_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_running_corrects += torch.sum(preds == labels.data)

            val_epoch_loss = val_running_loss / len(val_subset) 
            val_epoch_acc = val_running_corrects.double() / len(val_subset)
            fold_history['val_loss'].append(val_epoch_loss)
            fold_history['val_acc'].append(val_epoch_acc.item())
            
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_epoch_acc = val_epoch_acc.item() # 儲存最佳準確率
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                print(f"  (Early stopping triggered at Epoch {epoch+1})")
                break
        
        # 儲存此 Fold 的最佳驗證準確率
        fold_results.append(best_epoch_acc)
        all_fold_histories.append(fold_history)
        print(f"Fold {fold + 1} 最佳驗證準確率: {best_epoch_acc:.4f}")

    # K-Fold 結束後，印出平均結果
    print(f"\n--- K-Fold 交叉驗證完成 ---")
    avg_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    print(f"K-Fold 平均驗證準確率: {avg_accuracy:.4f} (標準差: {std_accuracy:.4f})")
    for i, acc in enumerate(fold_results):
        print(f"  - Fold {i+1} 準確率: {acc:.4f}")

    
    # --- 階段三：訓練最終模型 ---
    print("\n--- 階段三：訓練最終模型 (使用全部 80% 資料) ---")
    
    # 將使用全部 '80% 練習題' 來訓練最終模型
    final_train_loader = DataLoader(train_full_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=2, pin_memory=True, persistent_workers=True)

    # 建立一個全新的模型
    final_model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.classifier.parameters(), lr=LEARNING_RATE)

    # 根據 K-Fold 數據 (平均 18-20 epochs) 決定訓練 20 個 Epochs
    FINAL_EPOCHS = 20 
    print(f"將在全部 {len(train_full_dataset)} 張訓練資料上，訓練 {FINAL_EPOCHS} 個 Epochs...")
    
    final_model.train() # 確保在訓練模式
    for epoch in range(FINAL_EPOCHS):
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in final_train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_full_dataset)
        epoch_acc = running_corrects.double() / len(train_full_dataset)
        print(f"Final Train Epoch {epoch+1}/{FINAL_EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("最終模型訓練完成。")
    # 儲存最終模型
    final_model_path = os.path.join(OUTPUT_DIR, 'final_vgg16_model.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"最終模型已儲存至: {final_model_path}")


    # --- 階段四：最終評估  ---
    print(f"\n--- 階段四：在 {len(test_dataset)} 張 (20%) 最終測試集上評估 ---")
    
    # 建立測試集載入器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=2, pin_memory=True, persistent_workers=True)

    final_model.eval() # 設置為評估模式

    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())

    test_epoch_loss = test_loss / len(test_dataset)
    test_epoch_acc = test_corrects.double() / len(test_dataset)

    print(f"\n【最終測試集】總體評估:")
    print(f"  Test Loss:     {test_epoch_loss:.4f}")
    print(f"  Test Accuracy: {test_epoch_acc:.4f} ({(test_epoch_acc * 100):.2f}%)")

    # 分類報告
    print("\n【最終測試集】詳細分類報告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # 混淆矩陣
    print("正在繪製【最終測試集】混淆矩陣...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Final Test Set - Confusion Matrix (PyTorch)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_save_path = os.path.join(OUTPUT_DIR, 'final_test_set_confusion_matrix.png')
    plt.savefig(cm_save_path)
    print(f"最終測試集混淆矩陣圖已儲存至: {cm_save_path}")
    
    # 繪製 K-Fold 的訓練曲線
    plot_training_history(all_fold_histories, OUTPUT_DIR)

    print("\n--- 全部分析完成 (K-Fold) ---")
    print(f"所有輸出檔案 (模型、圖片) 皆已儲存在: {os.path.abspath(OUTPUT_DIR)}")


# ==============================================================================
if __name__ == '__main__':
    freeze_support()  
    main_process()