# 🪖 Helmet Detection for Bike Riders using YOLOv8

### 🚀 Final Project — Vadlapudi Varun Kumar

---

## 📌 Project Overview

Helmet violations are a major cause of two-wheeler fatalities. Manual monitoring is slow, error-prone, and cannot scale to large traffic systems.
This project implements an **automated Helmet vs No-Helmet detection system** using **YOLOv8**, capable of detecting bike riders and identifying whether they are wearing helmets.

---

## 🎯 Objective

To build a deep-learning-based system that detects bike riders and classifies them into:

✔ **Helmet**
❌ **No Helmet**

using image/video input.

---

## 🧠 Key Features

✔ Uses **YOLOv8** — a state-of-the-art object detection model
✔ Works on **images** (can be extended to video/CCTV)
✔ Detects rider + helmet status with bounding boxes
✔ Fast, scalable, and ready for deployment
✔ Fully implemented inside **Jupyter Notebook**

---

## 📂 Project Structure

```
📁 Helmet-Detection-Project
│── Helmet_Detection_Week-1.ipynb   # Main code
│── README.md                       # Project documentation
│── images/                         # Sample images (Helmet / No Helmet)
│── runs/detect/predict/            # Output detections (Generated after running)
```

---

## 🛠️ Tools & Technologies Used

| Category        | Tools                               |
| --------------- | ----------------------------------- |
| Language        | Python                              |
| IDE             | Jupyter Notebook                    |
| ML Framework    | YOLOv8 (Ultralytics), PyTorch       |
| Libraries       | OpenCV, NumPy, Matplotlib, Requests |
| Version Control | Git & GitHub                        |

---

## 🏗️ Methodology

### ✔ Step-1 : Data Setup

* Download or collect motorcycle rider images
* Create sample Helmet / No-Helmet dataset

### ✔ Step-2 : Model Selection

* Select YOLOv8n (pre-trained on COCO)

### ✔ Step-3 : Detection Pipeline

* Load model
* Run inference on input images
* Visualize bounding boxes + labels

### ✔ Step-4 : Result Export

* Save predictions in `/runs/detect/predict/`

---

## 🧾 Final Code (Core Section)

```python
from ultralytics import YOLO
import cv2, requests, os
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")   # Load YOLO model

# Download sample images
os.makedirs("samples", exist_ok=True)
urls = {
    "helmet": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "nohelmet": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
}

for name, url in urls.items():
    img_path = f"samples/{name}.jpg"
    open(img_path, "wb").write(requests.get(url).content)

# Detection + display
def detect(img):
    result = model(img)
    plt.imshow(cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

detect("samples/helmet.jpg")
detect("samples/nohelmet.jpg")
```

---

## 📸 Sample Output

✔ Rider detected
✔ Helmet status identified
✔ Screenshot shown in PPT/report

*(Add output images here in GitHub for better presentation)*

---

## 🧩 Results

* Successful detection of helmet & non-helmet riders
* YOLOv8 model achieved fast inference
* Code fully executed inside Jupyter Notebook
* Output ready for presentation

---

## 🏁 Conclusion

This project demonstrates an efficient helmet detection system using YOLOv8.
It can support **traffic police, surveillance systems, and smart city safety applications** by automatically identifying helmet rule violations.

---

## 🔮 Future Work

🔹 Train model on Indian traffic CCTV footage
🔹 Add number-plate recognition
🔹 Build a Streamlit / Flask web app
🔹 Deploy as live CCTV monitoring system

---

## 👨‍💻 Developed By

**Vadlapudi Varun Kumar**
B.Tech – AI & Data Science
GitHub: **[https://github.com/VARUN30C4](https://github.com/VARUN30C4)**

---


