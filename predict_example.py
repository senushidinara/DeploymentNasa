
import tensorflow as tf
import numpy as np
import pickle
import os

# --- Configuration ---
MODEL_PATH = 'task_risk_cnn_best_model.h5'
SCALER_PATH = 'scaler.pkl'
N_CHANNELS = 61
N_SAMPLES = 500
CLASS_MAP = {0: "Low Workload (SAFE)", 1: "Moderate Workload (CAUTION)", 2: "High Workload (RISK)"}

def preprocess_eeg_data(X_raw, scaler):
    # X_raw shape must be (N_epochs, N_channels, N_samples) -> (1, 61, 500)
    N_epochs, N_channels, N_samples = X_raw.shape
    
    # 1. Reshape for scaling: (N, Channels * Samples)
    X_reshaped = X_raw.reshape(N_epochs, -1)
    
    # 2. Scale the data
    X_scaled_flat = scaler.transform(X_reshaped)

    # 3. Reshape back: (N, Channels, Samples)
    X_scaled = X_scaled_flat.reshape(N_epochs, N_channels, N_samples)
    
    # 4. Transpose to CNN's required shape: (N, Samples, Channels)
    X_final = np.transpose(X_scaled, (0, 2, 1))
    return X_final

def predict_workload(new_data_raw):
    # Load assets
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"ERROR: Could not load model or scaler. Check paths/files.")
        print(e)
        return "ERROR"

    # Preprocess
    X_processed = preprocess_eeg_data(new_data_raw, scaler)
    
    # Predict
    predictions_proba = model.predict(X_processed, verbose=0)
    predicted_class = np.argmax(predictions_proba, axis=1)[0]
    
    # Output
    confidence = predictions_proba[0][predicted_class]
    predicted_label = CLASS_MAP[predicted_class]
    
    print("
--- Prediction Output ---")
    print(f"Raw Input Shape: {new_data_raw.shape}")
    print(f"Probabilities (Low, Moderate, High): {predictions_proba[0].round(3)}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.3f}")
    return predicted_label

# --- Execution Example ---
if __name__ == "__main__":
    print("--- Running Deployment Test ---")
    # Simulate receiving 1 second of new raw EEG data
    # (1 Epoch, 61 Channels, 500 Samples)
    sample_data = np.random.rand(1, N_CHANNELS, N_SAMPLES).astype(np.float32)
    
    # NOTE: This prediction will be random because the sample_data is random noise!
    # The engineer will replace this with real data.
    predict_workload(sample_data)
