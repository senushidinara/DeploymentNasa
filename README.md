# 🧠 NeuroLoad: Cognitive Workload Classification from Raw EEG Signals

This project presents a **comparative analysis for classifying cognitive workload** into three levels (Low, Moderate, High) using **EEG data**.  
We demonstrate the **superior performance of Deep Learning over traditional Machine Learning** for this task.

---

## 🥇 Key Results

A **1D Convolutional Neural Network (CNN)** significantly outperformed a Random Forest (RF) classifier baseline.

| Model Type | Input Data | Final Test Accuracy | Performance Gap |
| :--- | :--- | :--- | :--- |
| 1D CNN (Deep Learning) | Raw Time-Series (500, 61) | **94.2%** | **32.2%** |
| Random Forest (Traditional ML) | 427 Engineered Features | **62.0%** | N/A |

> The **32.2% performance gap** indicates that the raw temporal structure of EEG signals contains critical discriminative features not captured by standard statistical summaries.

---

## 🔬 Methodology

### 1D CNN Model (Automated Feature Extraction)

The CNN was designed for **end-to-end processing of raw EEG data**.

- **Input Shape:** `(500, 61)` → 500 time steps × 61 channels  
- **Core Architecture:** Sequential blocks of `Conv1D` (32, 64, 128 filters) + `MaxPooling1D` layers  
- **Training:** Adam optimizer, categorical cross-entropy loss, trained for 20 epochs  

### Random Forest Baseline (Hand-Engineered Features)

The traditional approach relied on **manually summarized features**.

- **Feature Engineering:** 7 statistical features per channel → Mean, Median, Std, Min, Max, IQR, Skewness  
- **Feature Vector Dimension:** `61 channels × 7 features = 427 dimensions`  
- **Model:** Random Forest classifier with 100 trees, max depth 15  

---

## 🚧 Challenge: Moderate Workload Ambiguity

The Random Forest model struggled with **Class 1 (Moderate Workload)**, achieving:

- **Recall:** 0.43  
- **F1-score:** 0.49  

| True Predicted Class | Class 0 (Low) | Class 1 (Moderate) | Class 2 (High) |
| :--- | :--- | :--- | :--- |
| Class 1 (Moderate) | 205 | **383** | 306 |

> Moderate workload is a **physiologically transitional state**, sharing characteristics with both Low and High classes.  
> The CNN excels due to **learning complex, non-linear boundaries** in high-dimensional raw EEG data.

---

## 🛠️ Future Work

1. **Generalizability:** Test CNN on different experimental paradigms & participants  
2. **Interpretability:** Apply Grad-CAM to visualize influential time periods and channels  
3. **Deployment:** Integrate into **edge computing** for real-time, low-latency workload monitoring

---

## 🚀 Getting Started

Clone the repository to explore:

- Training scripts for the **1D CNN architecture**  
- Feature engineering process for the **Random Forest baseline**  

---

## 🛠️ Built With

Python | Keras | TensorFlow | NumPy | Pandas | Matplotlib | ScikitLearn | EEG | CNN | RandomForestp
