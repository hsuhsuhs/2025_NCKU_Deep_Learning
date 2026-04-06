import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch, cv2, numpy as np
import os
from ultralytics import YOLO
from unet_model import UNet

class IndustrialGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("KSDD2 Defect Inspection System - P76141267")
        self.window.geometry("1100x650") 
        
        # 當前腳本所在的資料夾路徑 
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 載入模型 
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # 使用 os.path.join 組合相對路徑
        yolo_weight_path = os.path.join(self.base_dir, 'runs', 'detect', 'ksdd2_yolo_train', 'weights', 'best.pt')
        unet_weight_path = os.path.join(self.base_dir, 'runs', 'segment', 'ksdd2_unet_train', 'best_unet.pth')
        
        # 載入權重
        self.yolo = YOLO(yolo_weight_path)
        self.unet = UNet(n_channels=1, n_classes=1).to(self.device)
        self.unet.load_state_dict(torch.load(unet_weight_path, map_location=self.device))
        self.unet.eval()

        # UI 組件 - 按鈕區
        header = tk.Frame(window)
        header.pack(pady=10)
        tk.Button(header, text="1. Load Image", command=self.open_img, font=("Arial", 11)).pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="2. Inference", command=self.predict, font=("Arial", 11), bg="white").pack(side=tk.LEFT, padx=10)

        # 圖片顯示區框架
        self.img_frame = tk.Frame(window)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # 創建三個分欄與標題
        self.create_panel("Original", 0)
        self.create_panel("Detection", 1)
        self.create_panel("Segmentation", 2)

    def create_panel(self, title, col):
        frame = tk.Frame(self.img_frame)
        frame.grid(row=0, column=col, padx=15)
        tk.Label(frame, text=title, font=("Arial", 13)).pack(pady=5)
        panel = tk.Label(frame, bg="#333333")
        panel.pack()
        if col == 0: self.panel_orig = panel
        elif col == 1: self.panel_det = panel
        else: self.panel_seg = panel

    def open_img(self):
        self.path = filedialog.askopenfilename()
        if self.path:
            # 使用 np.fromfile 支援路徑中有中文字符
            self.img_cv = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_to_panel(self.img_cv, self.panel_orig)

    def predict(self):
        # Detection (YOLO)
        results = self.yolo(self.img_cv)[0]
        det_res = results.plot() 
        
        # Segmentation (U-Net)
        gray = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # 使用訓練時的 512x512 輸入大小
        inp = torch.from_numpy(cv2.resize(gray, (512, 512))/255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            out_prob = torch.sigmoid(self.unet(inp)) 
            prob_map = cv2.resize(out_prob.cpu().numpy()[0,0], (w, h)) 
            mask_bin = (prob_map > 0.5).astype(np.uint8) 
            
        seg_res = self.img_cv.copy()
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                x, y, nw, nh = cv2.boundingRect(cnt)
                
                # 動態計算該區域平均機率值 (Confidence Score)
                region_prob = prob_map[y:y+nh, x:x+nw]
                avg_conf = np.mean(region_prob[region_prob > 0.5]) if np.any(region_prob > 0.5) else 0.5
                
                # 智慧型標籤位置計算 (防止被右緣卡掉)
                label = f"defect {avg_conf:.2f}"
                label_w = 110  
                label_h = 25   
                
                text_x = x
                if x + label_w > w:
                    text_x = w - label_w - 5 
                
                text_y = y - 5
                bg_y1, bg_y2 = y - label_h, y
                if y - label_h < 0:
                    text_y = y + nh + 20
                    bg_y1, bg_y2 = y + nh, y + nh + label_h

                # 畫藍色方框 (BGR: 255, 0, 0)
                cv2.rectangle(seg_res, (x, y), (x + nw, y + nh), (255, 0, 0), 2)
                # 實心標籤背景
                cv2.rectangle(seg_res, (text_x, bg_y1), (text_x + label_w, bg_y2), (255, 0, 0), -1)
                # 寫上白字文字
                cv2.putText(seg_res, label, (text_x + 5, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 半透明藍色遮罩 (Mask)
        blue_mask = np.zeros_like(self.img_cv)
        blue_mask[mask_bin > 0] = [255, 0, 0]
        seg_res = cv2.addWeighted(seg_res, 0.7, blue_mask, 0.3, 0)

        # 更新介面圖片
        self.show_to_panel(det_res, self.panel_det)
        self.show_to_panel(seg_res, self.panel_seg)

    def show_to_panel(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 符合教授 PPT 比例的 320x480 縮放
        img_pil = Image.fromarray(cv2.resize(img_rgb, (320, 480)))
        tk_img = ImageTk.PhotoImage(img_pil)
        panel.config(image=tk_img)
        panel.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialGUI(root)
    root.mainloop()