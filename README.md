---
title: Real Time Object Detection YOLOv8
emoji: 🎯
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 Real-Time Object Detection using YOLOv8

A browser-based real-time object detection app powered by **YOLOv8n** (Ultralytics) and **Gradio**.
Detect objects from your webcam, uploaded images, or videos — all running in the browser.

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎥 **Live Webcam** | Real-time per-frame detection via browser camera |
| 📷 **Image Upload** | Detect objects in any uploaded photo |
| 🎬 **Video Processing** | Full annotated video output with all detections |
| 🎛️ **Confidence Slider** | Tune detection threshold from 10% to 90% |
| 🏷️ **80 COCO Classes** | Person, car, dog, chair, phone, and 75 more |
| 🖼️ **Visual Annotations** | Colored bounding boxes + corner accents + confidence labels |

## 🚀 Run Locally

```bash
git clone https://github.com/Jis-4/real-time-object-detection-using-YOLOv8.git
cd real-time-object-detection-using-YOLOv8
pip install -r requirements.txt
python app.py
```
Opens at http://localhost:7860

## 🤗 Deploy on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose Gradio as the SDK
3. Push this repo to the Space

## 🧠 How It Works

Input (webcam / image / video) → Preprocess → YOLOv8n inference → NMS → Draw boxes → Annotated output + stats

## 📦 Tech Stack

- Model: YOLOv8n — Ultralytics (6MB, 80 COCO classes)
- UI: Gradio 4.44 — Image / Webcam / Video tabs
- Vision: OpenCV + Pillow for annotation
- Runtime: PyTorch (CPU or CUDA auto-detected)

## 📄 License

MIT © 2026