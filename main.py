"""
StyleSense: Retail Fashion Sales Prediction Dashboard & ML Backend
MBA Data Science & AI Project: Neural Networks and Deep Learning

This file:
1. Implements 3 Machine Learning Models:
   - Model 1: Baseline Linear Regression
   - Model 2: ANN-1 (Shallow Neural Network: 32 ReLU)
   - Model 3: ANN-2 (Deep Regularized Neural Network: 64 -> 32 ReLU)
2. Trains them on synthetic retail data with zero data leakage.
3. Evaluates test set metrics (MAE, RMSE, R^2).
4. Serializes models, scalers, and metadata into a pickle file ('model.pkl').
5. Provides an HTTP web server & prediction API for cloud deployment (Procfile).
"""

import os
import sys
import math
import json
import pickle
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. CONSTANTS & DOMAIN HEURISTICS
# ==============================================================================
CATEGORIES = ['T-Shirts', 'Jeans', 'Dresses', 'Ethnic Wear', 'Footwear', 'Accessories']
CHANNELS = ['Website', 'Mobile App', 'Marketplace']

BASE_CATEGORY_DEMAND = {
    'T-Shirts': 160,
    'Jeans': 110,
    'Dresses': 105,
    'Ethnic Wear': 75,
    'Footwear': 85,
    'Accessories': 140
}

PRICE_RANGES = {
    'T-Shirts': (599, 1299),
    'Jeans': (1599, 3499),
    'Dresses': (1299, 2999),
    'Ethnic Wear': (1999, 5499),
    'Footwear': (1499, 3999),
    'Accessories': (399, 1199)
}

FEATURE_NAMES = [
    'Month',
    'Cat_T-Shirts', 'Cat_Jeans', 'Cat_Dresses', 'Cat_Ethnic_Wear', 'Cat_Footwear', 'Cat_Accessories',
    'Chan_Website', 'Chan_Mobile_App', 'Chan_Marketplace',
    'Price', 'Discount_Pct', 'Ad_Spend', 'Festival_Season', 'Influencer_Campaign', 'Email_Campaign', 'Stock_Level'
]

PICKLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

# ==============================================================================
# 2. DATASET GENERATION
# ==============================================================================
def generate_dataset(num_records=2000, seed=42):
    """
    Synthesize weekly fashion retail sales data governed by genuine econometric logic.
    """
    np.random.seed(seed)
    X_list = []
    y_list = []
    records = []

    for i in range(num_records):
        week_id = f"W{str(i + 1).zfill(4)}"
        month = int(np.random.randint(1, 13))
        category = str(np.random.choice(CATEGORIES))
        channel = str(np.random.choice(CHANNELS))

        p_min, p_max = PRICE_RANGES[category]
        step = 50
        min_steps = p_min // step
        max_steps = p_max // step
        price = float((np.random.randint(min_steps, max_steps + 1) * step) - 1)

        # Discount % (0 to 60)
        discounts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
        disc_weights = [0.10, 0.10, 0.12, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.03]
        discount_pct = float(np.random.choice(discounts, p=disc_weights))

        # Ad spend in thousands (0 to 180k)
        ad_spend = float(round(np.random.uniform(0, 180), 1))

        # Festival season: higher in Oct-Nov (Diwali) and Dec-Feb (Weddings)
        fest_prob = 0.75 if month in [10, 11, 12, 1, 2] else 0.15
        festival_season = 1 if np.random.random() < fest_prob else 0

        influencer = 1 if np.random.random() < 0.30 else 0
        email = 1 if np.random.random() < 0.40 else 0
        stock_level = int(np.random.randint(200, 850))

        # Demand formula
        b = BASE_CATEGORY_DEMAND[category]
        p_norm = (price - p_min) / (p_max - p_min + 1e-5)
        e_price = 1.05 - 0.25 * p_norm

        m_fest = 1.0
        if festival_season == 1:
            if category == 'Ethnic Wear':
                m_fest = 1.65
            elif category == 'Dresses':
                m_fest = 1.45
            elif category == 'Footwear':
                m_fest = 1.30
            else:
                m_fest = 1.20

        d_ratio = discount_pct / 100.0
        m_disc = 1.0 + 1.2 * d_ratio + 0.5 * (d_ratio ** 2)
        boost_ad = 24.0 * math.log(1.0 + ad_spend / 7.0)

        m_chan = {'Website': 1.0, 'Mobile App': 1.15, 'Marketplace': 1.25}[channel]
        m_inf = 1.18 if influencer == 1 else 1.0
        m_email = 1.10 if email == 1 else 1.0

        raw_demand = (b * e_price * m_fest * m_disc * m_chan * m_inf * m_email) + boost_ad
        noise = np.random.normal(1.0, 0.07)
        demand = round(raw_demand * noise)
        units_sold = int(min(max(5, demand), stock_level))

        # One-hot encoding
        cat_oh = [1.0 if c == category else 0.0 for c in CATEGORIES]
        chan_oh = [1.0 if ch == channel else 0.0 for ch in CHANNELS]

        feat_vector = [float(month)] + cat_oh + chan_oh + [
            price, discount_pct, ad_spend, float(festival_season),
            float(influencer), float(email), float(stock_level)
        ]

        X_list.append(feat_vector)
        y_list.append(units_sold)

        records.append({
            'Week_ID': week_id, 'Month': month, 'Category': category, 'Channel': channel,
            'Price': price, 'Discount_Pct': discount_pct, 'Ad_Spend': ad_spend,
            'Festival_Season': festival_season, 'Influencer_Campaign': influencer,
            'Email_Campaign': email, 'Stock_Level': stock_level, 'Units_Sold': units_sold
        })

    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), records

# ==============================================================================
# 3. MODEL TRAINING & PICKLING PIPELINE
# ==============================================================================
def train_and_pickle_models(save_path=PICKLE_FILE, test_split=0.20):
    """
    Train 3 ML models on the same split, compute metrics, and save to pickle file.
    """
    print("=" * 70)
    print(" StyleSense: Training 3 Machine Learning Models for Retail Sales")
    print("=" * 70)

    X, y, records = generate_dataset(num_records=2000, seed=42)
    n = len(X)
    test_count = int(n * test_split)
    train_count = n - test_count

    print(f"Dataset: {n} samples | Training: {train_count} | Test: {test_count}")

    X_train, X_test = X[:train_count], X[train_count:]
    y_train, y_test = y[:train_count], y[train_count:]

    # Standardization (fitted strictly on training set to avoid data leakage)
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # --------------------------------------------------------------------------
    # Model 1: Baseline Linear Regression
    # --------------------------------------------------------------------------
    print("\n[1/3] Training Baseline Linear Regression...")
    model_baseline = LinearRegression()
    model_baseline.fit(X_train_s, y_train_s)
    p_base_s = model_baseline.predict(X_test_s)
    p_base = np.maximum(0, scaler_y.inverse_transform(p_base_s.reshape(-1, 1)).flatten())

    mae_base = float(mean_absolute_error(y_test, p_base))
    rmse_base = float(math.sqrt(mean_squared_error(y_test, p_base)))
    r2_base = float(r2_score(y_test, p_base))
    print(f"      -> Linear Baseline: R2 = {r2_base:.3f} | MAE = {mae_base:.2f} | RMSE = {rmse_base:.2f}")

    # --------------------------------------------------------------------------
    # Model 2: ANN-1 (Shallow Neural Network: 32 ReLU)
    # --------------------------------------------------------------------------
    print("[2/3] Training ANN-1 (Shallow Neural Network: 32 ReLU)...")
    model_ann1 = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    model_ann1.fit(X_train_s, y_train_s)
    p_ann1_s = model_ann1.predict(X_test_s)
    p_ann1 = np.maximum(0, scaler_y.inverse_transform(p_ann1_s.reshape(-1, 1)).flatten())

    mae_ann1 = float(mean_absolute_error(y_test, p_ann1))
    rmse_ann1 = float(math.sqrt(mean_squared_error(y_test, p_ann1)))
    r2_ann1 = float(r2_score(y_test, p_ann1))
    print(f"      -> ANN-1 (Shallow): R2 = {r2_ann1:.3f} | MAE = {mae_ann1:.2f} | RMSE = {rmse_ann1:.2f}")

    # --------------------------------------------------------------------------
    # Model 3: ANN-2 (Deep Regularized Neural Network: 64 -> 32 ReLU)
    # --------------------------------------------------------------------------
    print("[3/3] Training ANN-2 (Deep Regularized Neural Network: 64 -> 32 ReLU)...")
    model_ann2 = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=250,
        alpha=0.01,  # L2 regularization
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    model_ann2.fit(X_train_s, y_train_s)
    p_ann2_s = model_ann2.predict(X_test_s)
    p_ann2 = np.maximum(0, scaler_y.inverse_transform(p_ann2_s.reshape(-1, 1)).flatten())

    mae_ann2 = float(mean_absolute_error(y_test, p_ann2))
    rmse_ann2 = float(math.sqrt(mean_squared_error(y_test, p_ann2)))
    r2_ann2 = float(r2_score(y_test, p_ann2))
    print(f"      -> ANN-2 (Deep):    R2 = {r2_ann2:.3f} | MAE = {mae_ann2:.2f} | RMSE = {rmse_ann2:.2f}")

    # Determine best model
    models_dict = {
        'baseline': {'model': model_baseline, 'name': 'Linear Regression Baseline', 'mae': mae_base, 'rmse': rmse_base, 'r2': r2_base},
        'ann1': {'model': model_ann1, 'name': 'ANN-1 (Shallow Neural Network)', 'mae': mae_ann1, 'rmse': rmse_ann1, 'r2': r2_ann1},
        'ann2': {'model': model_ann2, 'name': 'ANN-2 (Deep Regularized Network)', 'mae': mae_ann2, 'rmse': rmse_ann2, 'r2': r2_ann2}
    }

    best_key = max(models_dict.keys(), key=lambda k: models_dict[k]['r2'])
    best_name = models_dict[best_key]['name']
    improvement = ((mae_base - models_dict[best_key]['mae']) / mae_base) * 100

    print("\n" + "=" * 70)
    print(f" WINNER: {best_name}")
    print(f" Test R2: {models_dict[best_key]['r2']:.3f} | MAE: {models_dict[best_key]['mae']:.2f} units")
    print(f" Error reduction over linear baseline: {improvement:.1f}%")
    print("=" * 70)

    # Package into serializable pickle payload
    payload = {
        'baseline': models_dict['baseline'],
        'ann1': models_dict['ann1'],
        'ann2': models_dict['ann2'],
        'best_key': best_key,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'feature_names': FEATURE_NAMES,
        'categories': CATEGORIES,
        'channels': CHANNELS,
        'price_ranges': PRICE_RANGES,
        'train_samples': train_count,
        'test_samples': test_count
    }

    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)

    # Also save models.pkl as alias for flexibility
    alias_path = os.path.join(os.path.dirname(save_path), "models.pkl")
    with open(alias_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\n[+] Pickle file successfully saved to:\n    {save_path}\n    {alias_path}")
    return payload

# ==============================================================================
# 4. PREDICTION INFERENCE API
# ==============================================================================
def predict_demand(input_dict, model_payload=None):
    """
    Predict sales units and gross revenue given a scenario input dictionary.
    """
    if model_payload is None:
        if not os.path.exists(PICKLE_FILE):
            model_payload = train_and_pickle_models(PICKLE_FILE)
        else:
            with open(PICKLE_FILE, 'rb') as f:
                model_payload = pickle.load(f)

    month = int(input_dict.get('Month', 10))
    category = input_dict.get('Category', 'Jeans')
    channel = input_dict.get('Channel', 'Mobile App')
    price = float(input_dict.get('Price', 2499))
    discount_pct = float(input_dict.get('Discount_Pct', 25))
    ad_spend = float(input_dict.get('Ad_Spend', 60))
    festival = int(input_dict.get('Festival_Season', 1))
    influencer = int(input_dict.get('Influencer_Campaign', 1))
    email = int(input_dict.get('Email_Campaign', 0))
    stock_level = int(input_dict.get('Stock_Level', 500))

    cat_oh = [1.0 if c == category else 0.0 for c in CATEGORIES]
    chan_oh = [1.0 if ch == channel else 0.0 for ch in CHANNELS]

    feat = [float(month)] + cat_oh + chan_oh + [
        price, discount_pct, ad_spend, float(festival),
        float(influencer), float(email), float(stock_level)
    ]

    scaler_x = model_payload['scaler_x']
    scaler_y = model_payload['scaler_y']
    best_key = model_payload['best_key']
    best_model = model_payload[best_key]['model']

    feat_s = scaler_x.transform([feat])
    pred_s = best_model.predict(feat_s)
    raw_pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()[0]

    final_units = int(max(0, min(round(raw_pred), stock_level)))
    post_disc_price = price * (1.0 - discount_pct / 100.0)
    gross_revenue = round(final_units * post_disc_price)
    net_revenue = round(gross_revenue - (ad_spend * 1000))
    stock_util = round((final_units / max(1, stock_level)) * 100, 1)

    return {
        'model_used': model_payload[best_key]['name'],
        'predicted_units': final_units,
        'gross_revenue_inr': gross_revenue,
        'net_revenue_after_ads_inr': net_revenue,
        'stock_utilization_pct': stock_util,
        'stockout_risk': stock_util >= 98.0
    }

# ==============================================================================
# 5. HTTP SERVER & API HANDLER (FOR PROCFILE / CLOUD DEPLOYMENT)
# ==============================================================================
class StyleSenseHandler(SimpleHTTPRequestHandler):
    """
    Serves static assets (index.html, CSS, JS) and handles /api/predict JSON requests.
    """
    def __init__(self, *args, **kwargs):
        directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            status = {
                'status': 'online',
                'pickle_exists': os.path.exists(PICKLE_FILE),
                'models': ['Linear Regression', 'ANN-1 (32 ReLU)', 'ANN-2 (64->32 ReLU)']
            }
            self.wfile.write(json.dumps(status).encode('utf-8'))
            return
        elif parsed.path == "/api/predict":
            params = parse_qs(parsed.query)
            clean_params = {k: v[0] for k, v in params.items()}
            try:
                res = predict_demand(clean_params)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body) if body else {}
                res = predict_demand(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

def run_server(port=5000):
    """
    Start the HTTP server on the configured port.
    """
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, StyleSenseHandler)
    print(f"\n[+] StyleSense Web Server running at http://localhost:{port}/")
    print(f"    - Dashboard: http://localhost:{port}/")
    print(f"    - API:       http://localhost:{port}/api/predict")
    print("    Press Ctrl+C to stop the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="StyleSense Retail ML Backend & Server")
    parser.add_argument("--train", action="store_true", help="Force retrain models and update pickle file")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to bind server to")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on launch")
    args = parser.parse_args()

    # If pickle does not exist or --train is passed, train models and save pickle
    if args.train or not os.path.exists(PICKLE_FILE):
        train_and_pickle_models(PICKLE_FILE)
    else:
        print(f"[+] Loaded existing model pickle from: {PICKLE_FILE}")

    # Launch browser if running locally on default port
    if not args.no_browser and "PORT" not in os.environ:
        try:
            webbrowser.open(f"http://localhost:{args.port}/")
        except Exception:
            pass

    # Start the server (matches Procfile 'web: python main.py')
    run_server(args.port)

if __name__ == "__main__":
    main()
