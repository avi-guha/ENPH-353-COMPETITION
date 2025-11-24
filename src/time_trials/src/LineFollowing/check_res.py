import cv2
import os

data_dir = r'c:\Users\avigu\ENPH-353-COMPETITION\src\time_trials\src\LineFollowing\data\run_0\images'
if not os.path.exists(data_dir):
    print("Data directory not found")
else:
    files = os.listdir(data_dir)
    if files:
        img_path = os.path.join(data_dir, files[0])
        img = cv2.imread(img_path)
        if img is not None:
            print(f"Image shape: {img.shape}")
        else:
            print("Failed to read image")
    else:
        print("No images found")
