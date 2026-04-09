import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Load model once at startup ───────────────────────────────────────────────
model = YOLO("yolov8n.pt")   # downloads ~6MB automatically on first run

PALETTE = [
    (255,51,102),(255,102,51),(255,204,51),(51,255,102),(51,204,255),
    (204,51,255),(255,51,204),(51,255,204),(255,153,51),(51,102,255),
    (255,51,153),(153,255,51),(51,255,153),(153,51,255),(255,153,102),
    (102,255,51),(51,153,255),(255,102,153),(51,255,102),(102,51,255),
]

def color(cls_id): return PALETTE[cls_id % len(PALETTE)]

def draw_boxes(img_arr, results):
    img = img_arr.copy()
    names = model.names
    counts = {}
    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]
        col = color(cls_id)
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        cs = 12
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img,(cx,cy),(cx+dx*cs,cy),col,3)
            cv2.line(img,(cx,cy),(cx,cy+dy*cs),col,3)
        text = f"{label.upper()}  {conf*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = y1-6 if y1>20 else y1+th+6
        cv2.rectangle(img,(x1,ly-th-4),(x1+tw+8,ly+2),col,-1)
        cv2.putText(img,text,(x1+4,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)
        counts[label] = counts.get(label,0)+1
    if counts:
        lines = [f"• **{k}**: {v}" for k,v in sorted(counts.items())]
        summary = f"### {sum(counts.values())} object(s) detected\n" + "\n".join(lines)
    else:
        summary = "No objects detected."
    return img, summary

def detect_image(image, confidence):
    if image is None:
        return None, "Upload an image to run detection."
    arr = np.array(image)
    results = model(arr, conf=confidence, verbose=False)
    annotated, summary = draw_boxes(arr, results)
    return Image.fromarray(annotated), summary

def detect_frame(frame, confidence):
    if frame is None: return frame
    results = model(frame, conf=confidence, verbose=False)
    out, _ = draw_boxes(frame, results)
    return out

# ── UI ───────────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🎯 Real-Time Object Detection — YOLOv8

Detect **80 COCO object classes** using YOLOv8n (Ultralytics).
Upload an image **or** stream from your webcam. Adjust confidence to tune sensitivity.

> **Model:** YOLOv8n · **Backend:** Python + Ultralytics + PyTorch · **UI:** Gradio
"""

with gr.Blocks(
    title="YOLOv8 Object Detector",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Rajdhani"), "sans-serif"],
    ),
    css=".gradio-container{max-width:1100px!important} h1{letter-spacing:2px}",
) as demo:

    gr.Markdown(DESCRIPTION)

    conf_slider = gr.Slider(
        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
        label="Confidence Threshold",
        info="Lower = more detections. Higher = only confident ones.",
    )

    with gr.Tabs():
        with gr.Tab("📷 Image Upload"):
            with gr.Row():
                with gr.Column():
                    img_in  = gr.Image(type="pil", label="Upload Image", sources=["upload","clipboard"])
                    btn     = gr.Button("🔍 Run Detection", variant="primary", size="lg")
                with gr.Column():
                    img_out = gr.Image(type="pil", label="Result")
                    summary = gr.Markdown("Upload an image and click **Run Detection**.")
            btn.click(detect_image, [img_in, conf_slider], [img_out, summary])
            gr.Examples(
                examples=[
                    ["https://ultralytics.com/images/bus.jpg", 0.4],
                    ["https://ultralytics.com/images/zidane.jpg", 0.5],
                ],
                inputs=[img_in, conf_slider],
                outputs=[img_out, summary],
                fn=detect_image,
                cache_examples=False,
                label="Try example images",
            )

        with gr.Tab("🎥 Live Webcam"):
            gr.Markdown(
                "Click **Start** to begin real-time detection from your webcam.\n\n"
                "> Best in Chrome / Firefox on desktop."
            )
            webcam_in  = gr.Image(sources=["webcam"], streaming=True,
                                  label="Webcam", type="numpy")
            webcam_out = gr.Image(label="Detection Output", type="numpy")
            webcam_in.stream(
                detect_frame,
                inputs=[webcam_in, conf_slider],
                outputs=[webcam_out],
                time_limit=300,
                stream_every=0.07,
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## How it works

| Step | Detail |
|------|--------|
| **Model** | YOLOv8n — Nano, fastest YOLOv8 variant |
| **Classes** | 80 COCO classes: person, car, dog, bottle, chair... |
| **Inference** | Python · Ultralytics · PyTorch |
| **UI** | Gradio · Deployable on Hugging Face Spaces |

## Pipeline
```
Image / Webcam Frame
     -> resize to 640x640
YOLOv8n backbone (CSPDarknet)
     -> Neck (FPN + PAN)
     -> Detection Head: [cx, cy, w, h, cls x80]
     -> NMS (IoU threshold 0.45)
     -> Bounding boxes on canvas
```

## Source
GitHub: github.com/Jis-4/real-time-object-detection-using-YOLOv8
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
