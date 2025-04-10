import streamlit as st
import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # for enhanced visualizations
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Import your project code from your module (adjust the module name if needed)
from main_v2 import Autoencoder, load_credit_card_data, load_nsl_kdd_data, get_reconstruction_errors

# Set seaborn style for all plots
sns.set_style("whitegrid")

# ------------------------------
# Helper Functions to Load Saved Models
# ------------------------------
def load_saved_credit_card_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "creditcard_model.pth"
    params_path = "creditcard_hyperparams.pkl"
    scaler_path = "creditcard_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(params_path) and os.path.exists(scaler_path):
        with open(params_path, "rb") as f:
            best_params = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X, y, _ = load_credit_card_data("creditcard.csv")
        input_size = X.shape[1]
        model = Autoencoder(
            input_size,
            best_params["hidden_size"],
            best_params["bottleneck_size"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, X, y, scaler, best_params
    else:
        return None, None, None, None, None

def load_saved_nslkdd_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "nslkdd_model.pth"
    params_path = "nslkdd_hyperparams.pkl"
    scaler_path = "nslkdd_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(params_path) and os.path.exists(scaler_path):
        with open(params_path, "rb") as f:
            best_params = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_train, y_train, X_test, y_test, _ = load_nsl_kdd_data("KDDTrain+.txt", "KDDTest+.txt")
        input_size = X_test.shape[1]
        model = Autoencoder(
            input_size,
            best_params["hidden_size"],
            best_params["bottleneck_size"],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, X_test, y_test, scaler, best_params
    else:
        return None, None, None, None, None

# ------------------------------
# Utility Functions for Enhanced Visualization & Metrics
# ------------------------------
def plot_reconstruction_error(errors, threshold, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, alpha=0.6, color="skyblue", label="Reconstruction Errors")
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Reconstruction Error", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    return fig

def display_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics_text = (
        f"**Accuracy:** {acc:.4f}  \n"
        f"**Precision:** {prec:.4f}  \n"
        f"**Recall:** {rec:.4f}  \n"
        f"**F1 Score:** {f1:.4f}"
    )
    return metrics_text

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.set_page_config(page_title="EvoAuto Anomaly Detection", layout="wide")
st.title("EvoAuto – Evolving Autoencoders for Anomaly Detection")
st.markdown(
    """
This app demonstrates the anomaly detection pipeline using **pre-trained models** for:
- **Credit Card Fraud Detection**
- **NSL-KDD Anomaly Detection**

Explore the results through interactive tabs below.
    """
)

# Sidebar: Let the user choose the dataset.
dataset_choice = st.sidebar.radio(
    "Select Dataset", 
    ("Credit Card Fraud Detection", "NSL-KDD Anomaly Detection")
)

# Use Streamlit tabs to organize content.
tabs = st.tabs(["Overview", "Results", "About"])

with tabs[0]:
    st.header("Overview")
    st.write(
        """
Welcome to EvoAuto! This application demonstrates the performance of autoencoder-based anomaly detection.
The system was trained solely on normal (non-fraudulent) data, so anomalies produce higher reconstruction errors.
Below, you can view evaluation metrics and visualizations for each dataset.
        """
    )
    st.info("Select a dataset from the sidebar and click 'Run Experiment' to view the results.")

with tabs[1]:
    st.header("Results")
    if st.sidebar.button("Run Experiment"):
        if dataset_choice == "Credit Card Fraud Detection":
            st.subheader("Credit Card Fraud Detection (Using Saved Model)")
            model, X, y, scaler, best_params = load_saved_credit_card_model()
            if model is None:
                st.error("Saved Credit Card model not found. Please run your training pipeline.")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_tensor = torch.tensor(X, dtype=torch.float32)
                dataset = TensorDataset(X_tensor)
                loader = DataLoader(dataset, batch_size=128, shuffle=False)
                criterion = torch.nn.MSELoss()
                errors = get_reconstruction_errors(model, loader, criterion, device)
                threshold = np.percentile(errors, 95)
                st.write(f"**Reconstruction error threshold (95th percentile):** {threshold:.4f}")
                y_pred = (errors > threshold).astype(int)
                st.markdown(display_metrics(y, y_pred))
                cm_fig = plot_confusion_matrix(y, y_pred, title="Confusion Matrix - Credit Card")
                st.pyplot(cm_fig)
                error_fig = plot_reconstruction_error(errors, threshold, "Reconstruction Error Distribution - Credit Card")
                st.pyplot(error_fig)
                st.write(f"**Total number of fraud samples:** {np.sum(y)}")
        else:
            st.subheader("NSL-KDD Anomaly Detection (Using Saved Model)")
            model, X_test, y_test, scaler, best_params = load_saved_nslkdd_model()
            if model is None:
                st.error("Saved NSL-KDD model not found. Please run your training pipeline.")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                dataset = TensorDataset(X_tensor)
                loader = DataLoader(dataset, batch_size=128, shuffle=False)
                criterion = torch.nn.MSELoss()
                errors = get_reconstruction_errors(model, loader, criterion, device)
                threshold = np.percentile(errors, 95)
                st.write(f"**Reconstruction error threshold (95th percentile):** {threshold:.4f}")
                y_pred = (errors > threshold).astype(int)
                report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0)
                st.text(report)
                st.markdown(display_metrics(y_test, y_pred))
                cm_fig = plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix - NSL-KDD")
                st.pyplot(cm_fig)
                error_fig = plot_reconstruction_error(errors, threshold, "Reconstruction Error Distribution - NSL-KDD")
                st.pyplot(error_fig)

with tabs[2]:
    st.header("About This App")
    st.markdown(
        """
**EvoAuto** is a project that employs evolving autoencoder models for anomaly detection.  
The autoencoder is trained on normal data only so that anomalous data yields significantly higher reconstruction errors.
    
This UI provides:
- Evaluation metrics (accuracy, precision, recall, F1-score)
- A confusion matrix visualization
- Histograms of reconstruction errors

### Future Enhancements
- Further interactive visualizations and model interpretability
- Integration with live data streams
- Improved error analysis tools
        """
    )
