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
OUTPUT_DIR = os.path.join(BASE_WORKING_DIR, 'output_pytorch_kfold_MLP_Adjusted') 

# --- 超參數 ---
IMG_SIZE = (32, 32) 
BATCH_SIZE = 32
LEARNING_RATE = 1e-3 # 從 1e-4 提高
EPOCHS = 50 
TEST_SPLIT_RATIO = 0.2
K_FOLDS = 4 
PATIENCE = 10 

# --- 獲取類別名稱 ---
if not os.path.exists(ORIGINAL_DATA_DIR):
    print(f"警告：找不到原始資料路徑 {ORIGINAL_DATA_DIR}。")
    CLASS_NAMES = []
    NUM_CLASSES = 0
else:
    CLASS_NAMES = sorted([d for d in os.listdir(ORIGINAL_DATA_DIR) if os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, d))])
    NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================================
#  模型定義 (MLP)
 
class MLP(nn.Module):
    def __init__(self, num_classes=10): 
        super(MLP, self).__init__()
        
        #input_features 根據 IMG_SIZE = (32, 32) 自動計算
        input_features = 3 * IMG_SIZE[0] * IMG_SIZE[1]
        print(f"MLP 模型建立，輸入特徵數: {input_features}") # 提示
        
        # 定義 MLP 的層 
        self.layers = nn.Sequential(
            # 隱藏層 1: (3072 -> 512)
            nn.Linear(input_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), 
            
            # 隱藏層 2: (512 -> 256)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # 輸出層: (256 -> num_classes)
            nn.Linear(256, num_classes)
        )

    # 'forward' 定義資料如何流過模型
    def forward(self, x):
        x = torch.flatten(x, 1) #  將圖片資料攤平  
        x = self.layers(x) #  讓攤平的向量流過 'layers'
        return x

# ==============================================================================
#  輔助函式定義 

# split_data
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
        train_files = all_files[:split_index] 
        test_files = all_files[split_index:] 
        # 將檔案複製到新資料夾
        for f in train_files:
            shutil.copy(os.path.join(original_class_path, f), os.path.join(train_class_path, f))
        for f in test_files:
            shutil.copy(os.path.join(original_class_path, f), os.path.join(test_class_path, f))
    print("資料分割完成。")


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
    
    plt.suptitle('Model Training History (PyTorch K-Fold MLP - Adjusted)') 
    save_path = os.path.join(output_dir, 'training_history_curves_kfold_last_mlp_adjusted.png') 
    plt.savefig(save_path)
    print(f"最後一折的訓練曲線圖已儲存至: {save_path}")


# ==============================================================================
#  建立 MLP 模型
def create_model(device):
    
    model = MLP(num_classes=NUM_CLASSES) #  實例化 MLP 模型
    model = model.to(device)   #  將模型送上 GPU/CPU
    print("MLP (多層感知器) 模型建立完畢。")
    return model

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 階段一：資料準備 ---
    print("\n--- 階段一：資料準備 (8:2 分割) ---")
    
    split_data(ORIGINAL_DATA_DIR, TRAIN_DIR, TEST_DIR, TEST_SPLIT_RATIO)

    # 2. 影像轉換 
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(IMG_SIZE), # 調整為 32x32
            transforms.RandomRotation(30), # 隨機旋轉 30 度
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # 基於 32x32 裁切
            transforms.RandomHorizontalFlip(), # 隨機水平翻轉
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 標準化
        ]),

        'val': transforms.Compose([ 
            transforms.Resize(IMG_SIZE), # 調整為 32x32
            transforms.CenterCrop(IMG_SIZE), # 中心裁切 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 標準化
        ]),
    }

    # 3. 載入 80% TRAIN_DIR 和 20% TEST_DIR
    train_full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms['val']) 
    val_clean_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['val'])

    print(f"已載入 {len(train_full_dataset)} 張圖片 (80%) 用於 K-Fold 訓練/驗證。")
    print(f"已載入 {len(test_dataset)} 張圖片 (20%) 用於最終測試 (已鎖定)。")
    print(f"影像將被縮放為 {IMG_SIZE} 進行訓練。")

    # --- 階段二：K-Fold 交叉驗證 ---
    print(f"\n--- 階段二：開始 {K_FOLDS}-Fold 交叉驗證 ---")
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_fold_histories = []
    
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
        
        # 優化器使用已調整的 LEARNING_RATE 並訓練所有參數
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

         # 訓練此 Fold
        best_val_loss = float('inf')
        epochs_no_improve = 0
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_epoch_acc = 0.0

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
                best_epoch_acc = val_epoch_acc.item() 
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                print(f"  (Early stopping triggered at Epoch {epoch+1})")
                break
        
        fold_results.append(best_epoch_acc)
        all_fold_histories.append(fold_history)
        print(f"Fold {fold + 1} 最佳驗證準確率: {best_epoch_acc:.4f}")

    print(f"\n--- K-Fold 交叉驗證完成 ---")
    avg_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    print(f"K-Fold 平均驗證準確率: {avg_accuracy:.4f} (標準差: {std_accuracy:.4f})")
    for i, acc in enumerate(fold_results):
        print(f"  - Fold {i+1} 準確率: {acc:.4f}")

    
    # --- 階段三：訓練最終模型 ---
    print("\n--- 階段三：訓練最終模型 (使用全部 80% 資料) ---")
    
    final_train_loader = DataLoader(train_full_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=2, pin_memory=True, persistent_workers=True)

    final_model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

    # 增加最終訓練的 Epochs，因為 MLP 從零學習需要更多時間
    FINAL_EPOCHS = 35 
    print(f"將在全部 {len(train_full_dataset)} 張訓練資料上，訓練 {FINAL_EPOCHS} 個 Epochs...")
    
    final_model.train() 
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
    
    final_model_path = os.path.join(OUTPUT_DIR, 'final_mlp_model_adjusted.pth') 
    torch.save(final_model.state_dict(), final_model_path)
    print(f"最終模型已儲存至: {final_model_path}")


    # --- 階段四：最終評估  ---
    print(f"\n--- 階段四：在 {len(test_dataset)} 張 (20%) 最終測試集上評估 ---")
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=2, pin_memory=True, persistent_workers=True)

    final_model.eval() 

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

    print("\n【最終測試集】詳細分類報告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("正在繪製【最終測試集】混淆矩陣...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Final Test Set - Confusion Matrix (PyTorch MLP - Adjusted)') 
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_save_path = os.path.join(OUTPUT_DIR, 'final_test_set_confusion_matrix_mlp_adjusted.png') 
    plt.savefig(cm_save_path)
    print(f"最終測試集混淆矩陣圖已儲存至: {cm_save_path}")
    
    plot_training_history(all_fold_histories, OUTPUT_DIR)

    print("\n--- 全部分析完成 (MLP K-Fold - Adjusted) ---")
    print(f"所有輸出檔案 (模型、圖片) 皆已儲存在: {os.path.abspath(OUTPUT_DIR)}")


# ==============================================================================
if __name__ == '__main__':
    freeze_support()  
    main_process()