# ⚙️ Predictive Maintenance of Industrial Machines

> An end-to-end Machine Learning system that predicts equipment failure using real-time sensor data — Temperature, Vibration, Pressure, and Runtime.

---

## 📁 Project Structure

```
predictive-maintenance/
│
├── data/
│   ├── sensor_data.csv             # Raw synthetic dataset
│   └── sensor_data_cleaned.csv     # Cleaned dataset (generated after EDA)
│
├── model/
│   ├── random_forest_model.pkl     # Saved Random Forest model
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl                  # StandardScaler
│
├── notebooks/
│   └── predictive_maintenance.ipynb  # Full Jupyter walkthrough
│
├── plots/                           # Auto-generated plots
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── boxplots.png
│   ├── pairplot.png
│   ├── feature_importance.png
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   └── model_comparison.png
│
├── src/
│   ├── generate_data.py            # Synthetic dataset generator
│   ├── eda.py                      # Exploratory Data Analysis
│   └── train_model.py              # Model training & evaluation
│
├── app.py                          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Set up environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### Step 2 — Generate Dataset

```bash
python src/generate_data.py
```
Creates `data/sensor_data.csv` with 5,000 samples and ~10% failure rate.

### Step 4 — Train Models

```bash
python src/train_model.py
```
Trains Logistic Regression and Random Forest.  
Prints accuracy, ROC-AUC, classification reports.  
Saves models to `model/`.

### Step 5 — Launch Streamlit App

```bash
streamlit run app.py
```
Opens at **http://localhost:8501** in your browser.

---

## 🧪 Running in Jupyter Notebook

```bash
pip install notebook
jupyter notebook notebooks/predictive_maintenance.ipynb
```
Run all cells top-to-bottom for the full walkthrough.

---

## 🖥️ Running in VS Code

1. Install extensions: **Python**, **Pylance**, **Jupyter**
2. Open folder: `File → Open Folder → predictive-maintenance/`
3. Select interpreter: `Ctrl+Shift+P → Python: Select Interpreter`
4. Run scripts with ▶️ or via integrated terminal
5. For the Streamlit app: open terminal → `streamlit run app.py`

---

## 📊 Model Performance (Typical Results)

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~88%     | ~0.91   |
| Random Forest       | ~94%     | ~0.97   |

---

## 🔑 Key Features

| Feature         | Description                    |
|-----------------|--------------------------------|
| Temperature     | Machine operating temperature  |
| Vibration       | Vibration level in mm/s        |
| Pressure        | Operating pressure in bar      |
| Runtime Hours   | Cumulative machine runtime     |

---

## 📈 Power BI Dashboard Guidance

### Connect Data
- Import `data/sensor_data_cleaned.csv` into Power BI Desktop

### Recommended Visuals

| Visual Type        | What to Show                                      |
|--------------------|--------------------------------------------------|
| **KPI Cards**      | Total machines, failure count, failure rate %     |
| **Line Chart**     | Sensor readings over time (temperature trend)     |
| **Pie/Donut**      | Failure vs No-Failure ratio                       |
| **Scatter Plot**   | Temperature vs Vibration colored by failure       |
| **Bar Chart**      | Average sensor values by failure status           |
| **Gauge Chart**    | Current temperature/pressure vs safe threshold    |
| **Heatmap Table**  | Correlation values across all sensor features     |
| **Histogram**      | Distribution of runtime hours for failed machines |

### Key Insights to Highlight
1. At what temperature range do failures occur most?
2. Is there a runtime threshold beyond which failure probability spikes?
3. Which sensor combination (e.g., high temp + high vibration) is the strongest failure predictor?
4. Monthly failure trend — is it increasing or decreasing?

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualizations
- **Scikit-learn** — Machine Learning
- **Joblib** — Model persistence
- **Streamlit** — Web application

---

## 📄 License
MIT License — free to use, modify, and distribute.
