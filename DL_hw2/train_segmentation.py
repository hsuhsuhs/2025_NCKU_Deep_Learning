import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, os, numpy as np
import glob
import matplotlib.pyplot as plt
from unet_model import UNet

# 支援中文路徑的讀取函數
def cv_imread(path):
    # 使用 np.fromfile 確保在 Windows 中文路徑下也能正確讀取
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv_img

class KSDD2SegDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.img_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir = os.path.join(root_dir, mode, 'ground_truth')
        self.images_path = glob.glob(os.path.join(self.img_dir, "**", "*.png"), recursive=True)
        self.images = [os.path.relpath(f, self.img_dir) for f in self.images_path]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        rel_path = self.images[idx]
        img_path = os.path.join(self.img_dir, rel_path)
        possible_mask_paths = [
            os.path.join(self.mask_dir, rel_path),
            os.path.join(self.mask_dir, rel_path.replace('.png', '_GT.png')),
            os.path.join(self.mask_dir, rel_path.replace('.png', '_mask.png'))
        ]
        mask_path = None
        for p in possible_mask_paths:
            if os.path.exists(p):
                mask_path = p
                break
        img = cv_imread(img_path)
        if mask_path is not None:
            mask = cv_imread(mask_path)
        else:
            mask = np.zeros_like(img)
            
        if img is None:
            return torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512))
            
        img = cv2.resize(img, (512, 512)) / 255.0
        mask = cv2.resize(mask, (512, 512)) / 255.0

        # 確保 Mask 只有 0 與 1
        mask = (mask > 0.5).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()

def dice_coeff(pred, target):
    smooth = 1.
    p = (torch.sigmoid(pred) > 0.5).float()
    intersection = (p * target).sum()
    # 根據評分準則：2 * |Pred ∩ GT| / (|Pred| + |GT|)
    return (2. * intersection + smooth) / (p.sum() + target.sum() + smooth)

def train_seg():
    # 自動偵測設備
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 使用相對路徑 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'KSDD2')
    save_dir = os.path.join(base_dir, 'runs', 'segment', 'ksdd2_unet_train')
    os.makedirs(save_dir, exist_ok=True)
    
    print("正在初始化資料集...")
    dataset = KSDD2SegDataset(data_path, 'train')
    print(f"成功載入資料集！共 {len(dataset)} 張圖片。")

    # 設定 DataLoader
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = {'loss': [], 'dice': []}

    print(f"開始訓練...")
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_dice += dice_coeff(outputs, masks).item()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        history['loss'].append(avg_loss)
        history['dice'].append(avg_dice)
        print(f">>> Epoch {epoch} 完成! Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")

    # 繪製曲線圖並儲存 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['dice'], label='Dice Coefficient', color='orange')
    plt.title('Dice Coefficient (F1-Score)') # 標明 Dice 等於 F1-Score
    plt.xlabel('Epochs'); plt.legend()
    plt.savefig(os.path.join(save_dir, 'results_curve.png'))

    # 隨機樣本預測對比
    model.eval()
    with torch.no_grad():
        test_img, test_mask = next(iter(train_loader))
        pred = torch.sigmoid(model(test_img.to(device)))
        pred_mask = (pred > 0.5).float().cpu()
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(test_img[0,0], cmap='gray'); plt.title('Original Image')
        plt.subplot(1, 3, 2); plt.imshow(test_mask[0,0], cmap='gray'); plt.title('Ground Truth')
        plt.subplot(1, 3, 3); plt.imshow(pred_mask[0,0], cmap='gray'); plt.title('U-Net Prediction')
        plt.savefig(os.path.join(save_dir, 'val_samples.png'))

    # 儲存權重
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_unet.pth'))
    print(f"訓練完成！結果儲存於 {save_dir}")

if __name__ == '__main__':
    train_seg()