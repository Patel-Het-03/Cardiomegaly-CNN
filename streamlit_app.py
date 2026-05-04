import os
import json
from glob import glob

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np
import cv2  # ensure opencv-python installed
import sys
sys.path.append("src")

from gradcam import GradCAM
from gradcam_vis import overlay_cam_on_image

st.set_page_config(page_title="Cardiomegaly Prediction", layout="wide")

DATA_CSV = "data/Data_Entry_2017.csv"
HIST_JSON = "outputs/logs/history.json"
MODEL_PATH = "models/best_model.pt"
IMAGES_ROOT = "data/images"

@st.cache_resource
def load_model_and_cam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_PATH, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Target layer for ResNet: last conv block
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    return model, tf, cam, device
def plot_history():
    if not os.path.exists(HIST_JSON):
        st.info("No training history found. Train the model first (src/train.py).")
        return
    with open(HIST_JSON, "r") as f:
        hist = json.load(f)
    df = pd.DataFrame(hist)

    st.subheader("Training Curves")
    fig1 = plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    st.pyplot(fig1)

    for metric in ["val_acc", "val_precision", "val_recall", "val_f1"]:
        if metric in df.columns:
            fig = plt.figure()
            plt.plot(df["epoch"], df[metric])
            plt.xlabel("Epoch"); plt.ylabel(metric)
            st.pyplot(fig)

def eda_page():
    st.header("Dataset EDA (from CSV)")
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Cardiomegaly present?
    df["cardiomegaly"] = df["Finding Labels"].astype(str).str.contains("Cardiomegaly").astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Cardiomegaly +", f"{df['cardiomegaly'].sum():,}")
    col3.metric("Cardiomegaly rate", f"{df['cardiomegaly'].mean()*100:.2f}%")

    st.subheader("Label Distribution (Cardiomegaly)")
    fig = plt.figure()
    counts = df["cardiomegaly"].value_counts().sort_index()
    plt.bar(["0 (No)", "1 (Yes)"], counts.values)
    plt.ylabel("Count")
    st.pyplot(fig)

    st.subheader("Gender Distribution")
    if "Patient Gender" in df.columns:
        fig = plt.figure()
        g = df["Patient Gender"].value_counts()
        plt.bar(g.index.astype(str), g.values)
        plt.ylabel("Count")
        st.pyplot(fig)

    st.subheader("Age (rough)")
    if "Patient Age" in df.columns:
        # NIH ages are like "058Y" -> 58
        def parse_age(x):
            s = str(x)
            try:
                return int(s.replace("Y", ""))
            except:
                return np.nan
        df["age"] = df["Patient Age"].apply(parse_age)
        fig = plt.figure()
        plt.hist(df["age"].dropna(), bins=50)
        plt.xlabel("Age"); plt.ylabel("Count")
        st.pyplot(fig)

def inference_page():
    st.header("Cardiomegaly Detection + Grad-CAM")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train first: python src/train.py")
        return

    model, tf, cam, device = load_model_and_cam()

    uploaded = st.file_uploader("Upload a chest X-ray PNG/JPG", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to predict Cardiomegaly.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prediction
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(0).squeeze(0)
        prob = torch.sigmoid(logit).item()

    st.subheader("Prediction")
    st.write(f"**Cardiomegaly probability:** `{prob:.4f}`")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.write(f"**Predicted class:** {'Cardiomegaly' if prob >= thr else 'No Cardiomegaly'}")

    # Grad-CAM (needs gradients => no torch.no_grad here)
    st.subheader("Grad-CAM Explanation")

    alpha = st.slider("Heatmap strength (alpha)", 0.0, 1.0, 0.40, 0.05)

    # Make a second input with gradients enabled
    x_cam = tf(img).unsqueeze(0).to(device)
    x_cam.requires_grad_(True)

    cam_map = cam(x_cam)  # HxW in [0,1] at target layer resolution

    # Convert original image to uint8 for overlay
    rgb = np.array(img)  # HxWx3 uint8
    overlay = overlay_cam_on_image(rgb, cam_map, alpha=alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    with col2:
        # show heatmap alone
        heat = (cv2.resize(cam_map, (rgb.shape[1], rgb.shape[0])) * 255).astype(np.uint8)
        st.image(heat, caption="Grad-CAM mask (0-255)", use_container_width=True)
def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["EDA", "Training Curves", "Results", "Inference"])

    if page == "EDA":
        eda_page()
    elif page == "Training Curves":
        plot_history()
    elif page == "Results":
        results_page()
    else:
        inference_page()

def results_page():
    st.header("Test Results (ROC, AUC, Confusion Matrix)")

    path = "outputs/logs/test_eval_summary.json"
    if not os.path.exists(path):
        st.warning("Run: python src/eval.py to generate test results.")
        return

    with open(path, "r") as f:
        s = json.load(f)

    st.metric("ROC-AUC", f"{s['auc']:.4f}")
    st.metric("Best Threshold (by F1)", f"{s['best_threshold']:.2f}")
    st.write(f"**Best F1:** {s['best_f1']:.4f}  |  **Precision:** {s['best_precision']:.4f}  |  **Recall:** {s['best_recall']:.4f}")

    # ROC curve plot
    st.subheader("ROC Curve")
    fig = plt.figure()
    plt.plot(s["roc_fpr"], s["roc_tpr"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    st.pyplot(fig)

    # Confusion matrix
    st.subheader("Confusion Matrix (at best threshold)")
    cm = np.array(s["cm"])
    # cm = [[TN, FP],[FN, TP]]
    st.write("Rows: True (0,1), Cols: Pred (0,1)")
    st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

if __name__ == "__main__":
    main()