import os
# 1. 必須在最前面：解決 Mac MPS 不支援 NMS 算子的問題
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch, cv2, numpy as np
from ultralytics import YOLO
from unet_model import UNet

class IndustrialGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("KSDD2 Defect Inspection System - P76141267 (Mac)")
        self.window.geometry("1100x650") 
        
        # 2. 自動偵測 Apple Silicon 加速
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 獲取當前資料夾相對路徑
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 載入模型權重
        yolo_path = os.path.join(self.base_dir, 'runs', 'detect', 'ksdd2_yolo_train', 'weights', 'best.pt')
        unet_path = os.path.join(self.base_dir, 'runs', 'segment', 'ksdd2_unet_train', 'best_unet.pth')
        
        self.yolo = YOLO(yolo_path)
        self.unet = UNet(n_channels=1, n_classes=1).to(self.device)
        self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        self.unet.eval()

        # UI 組件
        header = tk.Frame(window)
        header.pack(pady=10)
        tk.Button(header, text="1. Load Image", command=self.open_img).pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="2. Inference", command=self.predict).pack(side=tk.LEFT, padx=10)

        self.img_frame = tk.Frame(window)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=10)
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
            # Mac 讀取影像優化
            self.img_cv = cv2.imread(self.path)
            if self.img_cv is None:
                self.img_cv = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.show_to_panel(self.img_cv, self.panel_orig)

    def predict(self):
        # ---------------------------------------------------------
        # Detection (YOLO) 
      
        img_rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
        results = self.yolo.predict(img_rgb, imgsz=640, conf=0.35, device=self.device, verbose=False)[0]
        
        # 準備畫布 (用原圖 BGR 格式，這樣顯示顏色才正常)
        det_res = self.img_cv.copy()
        
        # [Debug] 印出偵測數量，請看 Terminal 顯示多少
        print(f"DEBUG: YOLO detected {len(results.boxes)} boxes.")

        if len(results.boxes) > 0:
            for box in results.boxes:
                # 取得座標 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # [Debug] 印出座標，檢查是否合理
                print(f"Box: {x1}, {y1}, {x2}, {y2} | Conf: {conf:.2f}")

                # 畫藍色框 (BGR: 255, 0, 0)
                cv2.rectangle(det_res, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # 寫字
                label = f"defect {conf:.2f}"
                # 防止字跑到上面外面
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 25
                cv2.putText(det_res, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            # 如果沒抓到，在圖片上寫 No Detect
            cv2.putText(det_res, "No Defects Found", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---------------------------------------------------------
        # 2. Segmentation (U-Net) 
        gray = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
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
                region_prob = prob_map[y:y+nh, x:x+nw]
                avg_conf = np.mean(region_prob[region_prob > 0.5]) if np.any(region_prob > 0.5) else 0.5
                
                label = f"defect {avg_conf:.2f}"
                text_x = max(0, min(x, w - 115))
                text_y = y - 5 if y - 25 > 0 else y + nh + 20

                cv2.rectangle(seg_res, (x, y), (x + nw, y + nh), (255, 0, 0), 2)
                cv2.rectangle(seg_res, (text_x, text_y-20), (text_x + 110, text_y+5), (255, 0, 0), -1)
                cv2.putText(seg_res, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        blue_mask = np.zeros_like(self.img_cv)
        blue_mask[mask_bin > 0] = [255, 0, 0]
        seg_res = cv2.addWeighted(seg_res, 0.7, blue_mask, 0.3, 0)

        # 3. 顯示結果
        self.show_to_panel(det_res, self.panel_det)
        self.show_to_panel(seg_res, self.panel_seg)

    def show_to_panel(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 修正原本 NameError
        img_pil = Image.fromarray(cv2.resize(img_rgb, (320, 480)))
        tk_img = ImageTk.PhotoImage(img_pil)
        panel.config(image=tk_img)
        panel.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialGUI(root)
    root.mainloop()