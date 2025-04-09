import streamlit as st
import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

# Import your project code from my_project.py
from my_project import Autoencoder, load_credit_card_data, load_nsl_kdd_data, get_reconstruction_errors

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
        model = Autoencoder(input_size,
                            best_params["hidden_size"],
                            best_params["bottleneck_size"],
                            dropout_rate=best_params["dropout_rate"]).to(device)
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
        model = Autoencoder(input_size,
                            best_params["hidden_size"],
                            best_params["bottleneck_size"],
                            dropout_rate=best_params["dropout_rate"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, X_test, y_test, scaler, best_params
    else:
        return None, None, None, None, None

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.set_page_config(page_title="EvoAuto Anomaly Detection", layout="wide")
st.title("EvoAuto – Evolving Autoencoders for Anomaly Detection")
st.markdown("""
This app demonstrates the anomaly detection pipeline using **pre-trained models** for:
- **Credit Card Fraud Detection**
- **NSL-KDD Anomaly Detection**

If the required saved model files are not found, please run your training pipeline first.
""")

dataset_choice = st.sidebar.radio("Select Dataset", ("Credit Card Fraud Detection", "NSL-KDD Anomaly Detection"))
if st.sidebar.button("Run Experiment"):
    if dataset_choice == "Credit Card Fraud Detection":
        st.subheader("Credit Card Fraud Detection (Using Saved Model)")
        model, X, y, scaler, best_params = load_saved_credit_card_model()
        if model is None:
            st.error("Saved Credit Card model not found. Please run your training pipeline to generate the model, hyperparameters, and scaler.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tensor = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=128, shuffle=False)
            criterion = torch.nn.MSELoss()
            errors = get_reconstruction_errors(model, loader, criterion, device)
            threshold = np.percentile(errors, 95)
            st.write(f"**Reconstruction error threshold (95th percentile):** {threshold:.4f}")
            # For demonstration, we display the reconstruction error distribution.
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(errors, bins=50, alpha=0.6, label='Reconstruction Errors')
            ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
            ax.set_title("Reconstruction Error Distribution - Credit Card")
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
    else:
        st.subheader("NSL-KDD Anomaly Detection (Using Saved Model)")
        model, X_test, y_test, scaler, best_params = load_saved_nslkdd_model()
        if model is None:
            st.error("Saved NSL-KDD model not found. Please run your training pipeline to generate the model, hyperparameters, and scaler.")
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
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(errors, bins=50, alpha=0.6, label='Reconstruction Errors')
            ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
            ax.set_title("Reconstruction Error Distribution - NSL-KDD")
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

st.markdown("""
---
**Note:** If the saved model files are not found, please ensure you have run the training pipeline to generate:
- For Credit Card: `creditcard_model.pth`, `creditcard_hyperparams.pkl`, and `creditcard_scaler.pkl`
- For NSL-KDD: `nslkdd_model.pth`, `nslkdd_hyperparams.pkl`, and `nslkdd_scaler.pkl`
""")
