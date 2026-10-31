# StyleSense: Sales Prediction Dashboard for a Retail Fashion D2C Brand Using Artificial Neural Networks

An interactive, in-browser and Python-powered machine learning analytics dashboard designed for MBA Data Science & AI (*Neural Networks and Deep Learning* subject project).

![Project Status](https://img.shields.io/badge/Status-Complete-emerald?style=flat-square)
![Tech Stack](https://img.shields.io/badge/Tech-Python%20%7C%20Scikit--Learn%20%7C%20TensorFlow.js%20%7C%20Chart.js-indigo?style=flat-square)
![Deployment](https://img.shields.io/badge/Deployment-Procfile%20%7C%20Heroku%20%7C%20Render%20%7C%20GitHub%20Pages-blue?style=flat-square)

---

## 🌟 Overview & Business Objective
Modern Direct-to-Consumer (D2C) fashion brands face compressed seasonal cycles, volatile promotional elasticity, and substantial advertising acquisition costs. **StyleSense** models weekly category-level sales demand to help retail managers minimize:
- **Stockouts (Under-forecasting):** Missed revenue, customer churn, and brand degradation.
- **Overstocking (Over-forecasting):** High inventory holding costs and margin-eroding clearance markdowns.

---

## 🚀 Key Features

1. **Three Machine Learning Models in `main.py`**:
   - **Baseline Benchmark**: Linear Regression (`LinearRegression()`)
   - **ANN-1 (Shallow Network)**: Multi-Layer Perceptron with 1 hidden layer (`MLPRegressor(hidden_layer_sizes=(32,), activation='relu')`)
   - **ANN-2 (Deep Regularized Network)**: Deeper Multi-Layer Perceptron with 2 hidden layers (`MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01)`) with Early Stopping.
2. **Model Persistence via Pickle (`model.pkl` / `models.pkl`)**:
   - Serializes trained weights, scikit-learn standard scalers (`scaler_x`, `scaler_y`), feature mappings, and comparative test metrics ($R^2$, MAE, RMSE).
   - Allows instant prediction inference without retraining.
3. **Cloud Deployment Ready (`Procfile`)**:
   - Contains `web: python main.py` for one-click deployment on **Heroku**, **Render**, **Railway**, or **Koyeb**.
   - Built-in lightweight HTTP server and `/api/predict` REST endpoint.
4. **Interactive In-Browser Dashboard (`index.html`)**:
   - Client-side TensorFlow.js neural network training, real-time loss curves, executive KPI cards, 5 Chart.js visualizers, and What-If scenario simulations.
5. **Zero Data Leakage & Out-of-Sample Evaluation**:
   - Standardizers fitted strictly on training data (80%) and evaluated on unseen held-out test data (20%).

---

## 📊 Model Evaluation Benchmark (Test Set)

| Model | Architecture | Parameters / Neurons | Test $R^2$ | Test MAE | Test RMSE | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Baseline** | `LinearRegression()` | 18 weights | ~0.821 | ~21.20 units | ~27.45 units | Parsimonious Baseline |
| **ANN-1 (Shallow)** | `MLPRegressor(32)` | 32 ReLU | ~0.870 | ~17.79 units | ~23.38 units | Captures Non-linear Price Response |
| **ANN-2 (Deep Regularized)** | `MLPRegressor(64, 32)` | 64 $\to$ 32 ReLU | **~0.883** | **~17.71 units** | **~22.26 units** | **🏆 Best Model (16.5% error reduction)** |

---

## 📁 Repository Structure

```text
├── Procfile             <- Cloud deployment process definition (web: python main.py)
├── requirements.txt     <- Python dependencies (numpy, scikit-learn)
├── main.py              <- ML training pipeline, pickler, API & server
├── model.pkl            <- Pickled trained models, scalers, and metadata
├── models.pkl           <- Alias pickle file for easy import
├── index.html           <- Complete interactive single-file dashboard
├── data/                <- Sample synthetic dataset CSV exports
├── Front End/           <- Front-end components backup
└── README.md            <- Project documentation & viva guide
```

---

## 💻 How to Run

### Option 1: Run Python ML Pipeline & Server
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Retrain models & update pickle file
python main.py --train

# 3. Start web server & open dashboard (default port 5000)
python main.py
```

### Option 2: Run Directly in Browser (No Python Required)
Double-click `index.html` in Google Chrome or Microsoft Edge. All neural network training and visualizations run client-side via TensorFlow.js.

### Option 3: Deploy to Cloud (Heroku / Render)
1. Link this repository to your **Heroku** or **Render** account.
2. The platform automatically detects `Procfile` (`web: python main.py`) and `requirements.txt`.
3. The dashboard and `/api/predict` API are live instantly.

---

## 📜 License & Academic Integrity
Developed as an academic submission for MBA Data Science & Artificial Intelligence (*Neural Networks and Deep Learning* module). All synthetic formulas and architectural implementations are open for educational study.

**Author:** SOURAV VERMA  
**Roll No:** 25MDA10040
