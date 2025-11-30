import torch
import cv2
import numpy as np
import os
import pandas as pd
from model_vision import PilotNetVision

def visualize():
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PilotNetVision().to(device)
    model_path = 'best_model_vision_angular.pth'
    
    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Data
    data_dir = 'data'
    all_data = []
    for root, dirs, files in os.walk(data_dir):
        if 'log.csv' in files:
            try:
                df = pd.read_csv(os.path.join(root, 'log.csv'))
                df['base_path'] = root
                all_data.append(df)
            except: pass
            
    if not all_data:
        print("No data found")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Pick random samples
    indices = np.random.choice(len(df), 20)
    
    print(f"{'Image':<30} | {'True w':<10} | {'Pred w':<10} | {'Diff':<10}")
    print("-" * 70)
    
    for idx in indices:
        row = df.iloc[idx]
        img_path = os.path.join(row['base_path'], row['image_path'])
        image = cv2.imread(img_path)
        
        if image is None: continue
        
        # Preprocess
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        input_img = cv2.resize(input_img, (120, 120))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_w = output[0][0].item() * 3.0 # Denormalize (single output)
            
        true_w = row['w']
        diff = pred_w - true_w
        
        print(f"{os.path.basename(img_path):<30} | {true_w:6.2f}     | {pred_w:6.2f}     | {diff:6.2f}")
        
        # Draw on image
        display_img = cv2.resize(image, (400, 400))
        
        # Draw True (Green)
        cv2.putText(display_img, f"True: {true_w:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw Pred (Blue)
        cv2.putText(display_img, f"Pred: {pred_w:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Prediction", display_img)
        key = cv2.waitKey(0)
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()
