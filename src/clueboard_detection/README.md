## clueboard_detection Directory

### A directory containing materials:
1. RoboFlow image collection
2. RoboFlow dataset
3. YOLOv8 model training
4.  Trained YOLOv8 model
5.  YOLO inference results image collection
6.  Scripts for:
    1. CNN usage class
    2. Running CNN model on YOLO results
    3. YOLO inference node
    4. YOLO integration
7. A copy of trained clueboard reader CNN
   

General structure:
```
fizzer@skynet:~/ENPH-353-COMPETITION/src/clueboard_detection 2025-11-28 14:31:17
$ tree -L 2
.
├── clueboard_reader_CNN.h5
├── CMakeLists.txt
├── images
│   ├── img_10.png
│   └── ...
├── package.xml
├── README.md
├── roboflow_data_v1
│   ├── data.yaml
│   ├── runs
│   ├── train
│   ├── valid
│   └── yolov8n.pt
├── roboflow_data_v2
│   ├── 353-Clueboard-Detection.v2i.yolov8.zip
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── train
│   └── valid
├── runs
│   └── detect
├── src
│   ├── board_reader.py
│   ├── test_model.py
│   └── yolo_inference_live.py
├── yolo_inference_images
│   ├── img_0.png
│   └── ...
└── yolov8n.pt
```

Trained, best YOLO model saved in: `clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt`





