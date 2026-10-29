# StyleSense: Sales Prediction Dashboard for a Retail Fashion D2C Brand Using Artificial Neural Networks

An interactive, in-browser machine learning and analytics dashboard designed for MBA Data Science & AI (*Neural Networks and Deep Learning* subject project).

![Project Status](https://img.shields.io/badge/Status-Complete-emerald?style=flat-square)
![Tech Stack](https://img.shields.io/badge/Tech-TensorFlow.js%20%7C%20Chart.js%20%7C%20HTML5-indigo?style=flat-square)
![Deployment](https://img.shields.io/badge/Deployment-Zero--Install%20(Single%20File)-blue?style=flat-square)

---

## 🌟 Overview & Business Objective
Modern Direct-to-Consumer (D2C) fashion brands face compressed seasonal cycles, volatile promotional elasticity, and substantial advertising acquisition costs. **StyleSense** models weekly category-level sales demand to help retail managers minimize:
- **Stockouts (Under-forecasting):** Missed revenue, customer churn, and brand degradation.
- **Overstocking (Over-forecasting):** High inventory holding costs and margin-eroding clearance markdowns.

---

## 🚀 Key Features

1. **Executive Overview Dashboard**: Real-time KPI summary (Total Volume, Gross Revenue in ₹ INR, Average Markdown %, Average Ad Spend, Top Category) with 5 interactive visualizers (Monthly Seasonality, Category Breakdown, Channel Share, Discount Buckets, and Ad Saturation scatter plot).
2. **Synthetic Data Engine**: Generates 1,000 to 3,000 realistic weekly transaction records governed by genuine econometric relationships:
   - Base demand across 6 apparel categories (T-Shirts, Jeans, Dresses, Ethnic Wear, Footwear, Accessories)
   - Non-linear promotional discount elasticity ($1.2 \cdot D + 0.5 \cdot D^2$)
   - Logarithmic diminishing marginal returns on ad spend ($24 \cdot \ln(1 + \text{Ad}/7)$)
   - Indian festive surges (Diwali and Wedding seasons with up to +65% uplift in Ethnic Wear)
   - Channel differences (Website, Mobile App, Marketplaces)
   - Physical inventory caps (`Stock_Level`)
3. **In-Browser Neural Network Training (TensorFlow.js)**:
   - **Baseline Benchmark**: Linear Regression (`Dense(1, linear)`)
   - **ANN-1 (Shallow Network)**: `Dense(32, ReLU) -> Dense(1, linear)`
   - **ANN-2 (Deep Regularized Network)**: `Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, linear)`
   - Optimization: **Adam** optimizer, **Mean Squared Error (MSE)** loss, and **Early Stopping** (patience = 10 epochs).
   - Real-time training vs. validation loss tracking.
4. **Out-of-Sample Evaluation (Held-out Test Split)**:
   - Zero data leakage: scalers fitted strictly on training partition.
   - Evaluates test **MAE**, **RMSE**, and **$R^2$** scores.
   - Actual vs. Predicted scatter plot with dashed $y = x$ reference line.
   - Error residual frequency distribution histogram.
   - CSV export of test predictions.
5. **Managerial What-If Scenario Predictor**:
   - Interactive simulation form: vary category, channel, price, discount %, ad spend, festival flag, and stock level.
   - Computes forecasted demand, gross revenue, net revenue after advertising, and inventory utilization risk.
   - Generates an interactive **Discount Sensitivity Curve** (0% to 60%) highlighting optimal pricing thresholds.
6. **Academic Viva Guide**:
   - Comprehensive theoretical reference covering neuron math, forward propagation, ReLU non-linearity, MSE loss, Backpropagation, Adam optimization, Dropout regularization, and future DL roadmaps (CNNs, LLMs, LSTMs).

---

## 🛠️ Tech Stack & Architecture
- **Single File Delivery**: `index.html` houses all structure, CSS styling, and JavaScript logic.
- **TensorFlow.js (`v4.18.0`)**: In-browser hardware acceleration (WebGL / CPU) for tensor computations.
- **Chart.js (`v4.4.2`)**: Interactive data visualization canvas rendering.
- **No Backend / No Python Setup**: Operates entirely client-side; double-click to launch in any browser.

---

## 💻 How to Run

### Method 1: Local Browser (Fastest)
1. Clone or download this repository.
2. Double-click `index.html` to open it in **Google Chrome**, **Microsoft Edge**, **Brave**, or **Safari**.
*(An internet connection is required initially to load the CDN scripts).*

### Method 2: GitHub Pages (Free Live Web Hosting)
1. In your GitHub repository, go to **Settings** $\to$ **Pages**.
2. Under **Branch**, select `main` and root `/` folder, then click **Save**.
3. Your dashboard will be live at:
   `https://souravverma0038.github.io/-Fashion-Retail-Sales-Predictor/`

---

## 📊 Model Architecture Summary

| Model | Architecture | Parameters | Key Characteristic |
| :--- | :--- | :---: | :--- |
| **Linear Baseline** | `Input(17) -> Dense(1, Linear)` | 18 | Occam's Razor linear benchmark |
| **ANN-1 (Shallow)** | `Input(17) -> Dense(32, ReLU) -> Dense(1, Linear)` | 609 | Captures non-linear price & promo response |
| **ANN-2 (Deep Regularized)** | `Input(17) -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, Linear)` | 3,265 | Captures multi-factor synergies; Dropout prevents overfitting |

---

## 📜 License & Academic Integrity
Developed as an academic submission for MBA Data Science & Artificial Intelligence (*Neural Networks and Deep Learning* module). All synthetic formulas and architectural implementations are open for educational study.
