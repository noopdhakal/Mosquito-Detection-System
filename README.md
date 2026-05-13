# Mosquito-Detection-System
The project focuses on a computer vision based mosquito detection system that utilizes YOLO v5 to identify six specific mosquito species.

## 1. Create a virtual environment 
```bash
conda create -n mosquito python=3.11 -y
```

## 2. Activate the virtual environment
```bash
conda activate mosquito
```

## 3. Install the required dependencies
```bash
pip install -r requirements.txt

# 🦟 Mosquito Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-green)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange)

An end-to-end AI-powered mosquito detection system that detects and classifies mosquito species from images and webcam streams using YOLOv5. The project includes model training, evaluation, a Flask web API, a Bootstrap frontend, Docker containerization, and AWS deployment.

---

## 📌 Overview

Mosquito-borne diseases such as Dengue, Malaria, Chikungunya, and Zika continue to pose major public health challenges. Accurate identification of mosquito species is essential for effective vector surveillance and disease prevention.

This project automates mosquito detection using deep learning, enabling:

* Real-time image-based detection
* Webcam inference
* Species classification
* Automated deployment to the cloud

---

## ❗ Problem Statement

Manual mosquito classification is:

* Time-consuming
* Error-prone
* Dependent on trained experts
* Difficult to scale in field settings

This system addresses these limitations by using computer vision to detect mosquito species automatically.

---

## 🎯 Objectives

* Build a custom mosquito detection dataset.
* Train a YOLOv5 object detection model.
* Evaluate performance using Precision, Recall, and mAP.
* Develop a Flask REST API for inference.
* Create a web interface for image upload and webcam detection.
* Containerize the application using Docker.
* Deploy to AWS with GitHub Actions CI/CD.

---

## ✨ Features

* 🦟 Multi-class mosquito species detection
* 🖼️ Image upload prediction
* 🎥 Real-time webcam detection
* 📦 Flask REST API
* 🌐 Bootstrap-based frontend
* 🐳 Docker support
* ☁️ AWS deployment (ECR + EC2)
* 🔄 GitHub Actions CI/CD

---

## 📊 Dataset

The dataset was obtained from Roboflow Universe.

**Dataset Link:** [https://universe.roboflow.com/mosquitos-u6ipx/mosquito-detection-dataset/browse](https://universe.roboflow.com/mosquitos-u6ipx/mosquito-detection-dataset/browse)

### Dataset Details

* Annotated images in YOLO format
* 6 mosquito classes
* Train / Validation / Test split
* Bounding box annotations

---

## 🛠️ Tech Stack

| Category         | Tools                            |
| ---------------- | -------------------------------- |
| Computer Vision  | YOLOv5, OpenCV, PyTorch          |
| Backend          | Flask                            |
| Frontend         | HTML, CSS, Bootstrap, JavaScript |
| Dataset Platform | Roboflow                         |
| Containerization | Docker                           |
| CI/CD            | GitHub Actions                   |
| Cloud            | AWS ECR, EC2                     |
| Version Control  | Git, GitHub                      |

---

## 🏗️ System Architecture

```text
User Image / Webcam
        ↓
Bootstrap Frontend
        ↓
Flask REST API
        ↓
YOLOv5 Model
        ↓
Bounding Boxes + Labels + Confidence
        ↓
Annotated Output
```

---

## 📁 Project Structure

```text
mosquito-detection-system/
│── data/
│── notebooks/
│   └── mosquito_detection_training_yolov5.ipynb
│── yolov5/
│── static/
│── templates/
│── uploads/
│── outputs/
│── app.py
│── requirements.txt
│── Dockerfile
│── docker-compose.yml
│── .github/workflows/deploy.yml
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mosquito-detection-system.git
cd mosquito-detection-system
```

### 2. Create a Virtual Environment

```bash
conda create -n mosquito python=3.11
conda activate mosquito
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

```bash
python train.py \
  --img 416 \
  --batch 16 \
  --epochs 100 \
  --data data.yaml \
  --cfg ./models/custom_yolov5s.yaml \
  --weights yolov5s.pt \
  --name yolov5s_results \
  --cache
```

### Training Configuration

| Parameter  |     Value |
| ---------- | --------: |
| Image Size | 416 × 416 |
| Batch Size |        16 |
| Epochs     |       100 |
| Model      |   YOLOv5s |
| Classes    |         6 |

---

## 📈 Model Performance

The following metrics were extracted from the final epoch (`epoch 99`) of the training run.

| Metric              |    Score |
| ------------------- | -------: |
| Precision           | 0.952860 |
| Recall              | 0.940300 |
| mAP@0.5             | 0.974990 |
| mAP@0.5:0.95        | 0.720960 |
| Train Box Loss      | 0.025562 |
| Validation Box Loss | 0.021929 |

### Interpretation

* **Precision (95.29%)**: Most detections are correct.
* **Recall (94.03%)**: The model detects most mosquitoes present.
* **mAP@0.5 (97.50%)**: Excellent detection performance.
* **mAP@0.5:0.95 (72.10%)**: Strong localization performance under stricter IoU thresholds.

---

## 📉 Training Artifacts

The training process generated:

* `results.png` – training curves
* `results.csv` – epoch-by-epoch metrics
* `PR_curve.png` – precision-recall curve
* `F1_curve.png` – F1 score curve
* `confusion_matrix.png`
* `weights/best.pt`
* `weights/last.pt`

### Add Visualizations

```markdown
![Training Results](assets/results.png)
![Confusion Matrix](assets/confusion_matrix.png)
![PR Curve](assets/PR_curve.png)
```

---

## 🔍 Run Inference

```bash
python detect.py \
  --weights runs/train/yolov5s_results/weights/best.pt \
  --img 416 \
  --conf 0.25 \
  --source test/images
```

---

## 🧪 Evaluate on Test Set

```bash
python val.py \
  --weights runs/train/yolov5s_results/weights/best.pt \
  --data data.yaml \
  --task test \
  --img 416 \
  --verbose
```

This command reports per-class metrics for all six mosquito species.

---

## 🌐 Flask API Endpoints

| Method | Endpoint   | Description                 |
| ------ | ---------- | --------------------------- |
| GET    | `/`        | Web interface               |
| POST   | `/predict` | Predict from uploaded image |
| POST   | `/webcam`  | Real-time webcam detection  |
| GET    | `/health`  | Health check                |

---

## 🐳 Docker Usage

```bash
docker build -t mosquito-detector .
docker run -p 5000:5000 mosquito-detector
```

---

## 🔄 CI/CD Pipeline

1. Push code to GitHub.
2. GitHub Actions builds Docker image.
3. Push image to AWS ECR.
4. Deploy to EC2.

---

## ☁️ AWS Deployment

Services used:

* Amazon Elastic Container Registry (ECR)
* Amazon Elastic Compute Cloud (EC2)

Deployment workflow:

```text
GitHub → GitHub Actions → Docker Build → AWS ECR → EC2
```

---

## 🌍 Applications

* Public health surveillance
* Mosquito species monitoring
* Disease prevention research
* Smart mosquito traps
* Educational demonstrations

---

## 🚀 Future Improvements

* Expand the dataset with more species.
* Upgrade to YOLOv8.
* Build a mobile application.
* Add analytics dashboard.
* Integrate with IoT-based mosquito traps.

---

## 📚 References

* YOLOv5 Documentation: [https://docs.ultralytics.com/yolov5/](https://docs.ultralytics.com/yolov5/)
* Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
* OpenCV Documentation: [https://opencv.org/](https://opencv.org/)
* Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)
* AWS Documentation: [https://docs.aws.amazon.com/](https://docs.aws.amazon.com/)

---

## 📄 License

This project is licensed under the MIT License.
