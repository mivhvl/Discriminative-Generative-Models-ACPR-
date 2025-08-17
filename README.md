# 🤖 Face Generator and Fakes Discriminator – FEUP

This project involves developing two models for human deepfakes: a **GAN** to generate realistic fake images and a **classifier** to distinguish real from fake faces. It summarizes our iterative development, experiments, evaluations, and conclusions.

---

## 📁 Project Structure
```
Root/
├── DLR-Group6.pdf             # Project report
├── DLR-Discriminator.py       # Classifier notebook / script to distinguish real vs fake faces
├── DLR.py                     # GAN training and generation
├── _augmentation.py           # Data augmentation utilities
├── _preprocess.py             # Data preprocessing functions
├── _eda.py                    # Exploratory Data Analysis functions
├── _model.py                  # GAN models definitions
├── _fid.py                    # FID score calculation for GAN evaluation
├── yolov8n-face-lindvs.pt     # Pretrained YOLOv8 model for face detection
├── requirements.txt           # Dependencies
├── README.txt                 # This file
└── .gitignore                 # Git ignore list

```

## 📦 Dependencies

To install all required packages:
```
pip install -r requirements.txt
```
Minimum required packages:

- numpy
- tensorflow
- matplotlib

---

## 📘 Raport

Raport with the project details and results available in file `DRL_Group6.pdf`.

---

## 👥 Authors & Institution

- Author(s): Michal Kowalski, Pedro Azevedo, Pedro Pereira 
- Course: Deep and Reinforcement Learning
- Institution: FEUP – Faculdade de Engenharia da Universidade do Porto
- Date: June 2025

---
