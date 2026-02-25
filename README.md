# Auto MPG Prediction — Deep Neural Network Regression

**Course:** AI Programming — Winter 2026 | Ontario Tech University  
**Tools:** Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib  
**Task:** Regression (Predicting vehicle fuel efficiency in miles per gallon)

---

## Project Overview

This project builds a deep neural network to predict the fuel efficiency (MPG) of vehicles using the [UCI Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/Auto+MPG). The dataset covers vehicles manufactured between 1970 and 1982, and the model is ultimately used to predict MPG for hypothetical 2025 vehicles — demonstrating both the model's strengths and the limits of extrapolation beyond the training distribution.

---

## Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 392 (after dropping rows with missing values) |
| Features | 6 (Cylinders, Displacement, Horsepower, Weight, Acceleration, Year) |
| Target | MPG (continuous — miles per gallon) |
| Split | 50% Train / 50% Validation |
| Missing Data | Horsepower missing for 6 rows — filled with per-cylinder-group mean |

**Preprocessing steps:**
- Dropped `name` and `origin` columns (non-numeric / not relevant)
- Filled missing horsepower values using cylinder-group averages
- Converted 2-digit model years to 4-digit (e.g. 76 → 1976)
- Applied StandardScaler to all features

---

## Model Architecture

```
Input (6 features)
    ↓
Dense(128) + He-normal init + ReLU
    ↓
BatchNormalization + Dropout(0.2)
    ↓
Dense(64) + He-normal init + ReLU
    ↓
BatchNormalization + Dropout(0.2)
    ↓
Dense(32) + ReLU
    ↓
Dense(1)  — MPG output (linear activation)
```

**Optimizer:** Adam  
**Loss:** Mean Squared Error (MSE)  
**Metric:** Mean Absolute Error (MAE)  
**Training:** Up to 200 epochs with EarlyStopping (patience=20, restore best weights)

---

## Results

The model achieves a validation MAE of approximately **2–3 MPG** on the held-out validation set, which is competitive for this dataset given the 50/50 split and the age of the training data.

---

## Predicting 2025 Vehicles

The trained model was applied to 10 hypothetical 2025 vehicles with a range of engine configurations:

| Vehicle Profile | Predicted MPG |
|---|---|
| 6-cyl, 502 HP, moderate weight | ~12–15 mpg |
| 12-cyl, 730 HP, heavy | ~8–11 mpg |
| 8-cyl, 986 HP, 2025 | ~9–12 mpg |
| 4-cyl, small displacement, 2025 | ~18–22 mpg |

**Note on extrapolation:** The model was trained on 1970–1982 vehicles with horsepower typically between 50–230 HP. Predictions for modern high-performance vehicles (500+ HP) should be interpreted cautiously — the model is extrapolating beyond its training distribution. This is an intentional part of the analysis, illustrating a key limitation of supervised ML models.

---

## Project Structure

```
auto-mpg-regression/
│
├── auto.ipynb             # Main notebook with full analysis
├── README.md              # This file
└── images/
    └── mpg-training.png   # MSE loss and MAE curves
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/oumasseabderrahman1/auto-mpg-regression
cd auto-mpg-regression

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib

# Launch the notebook
jupyter notebook auto.ipynb
```

The dataset is loaded directly from the UCI repository — no manual download required.

---

## AI Assistance Disclosure

Portions of this code were developed with the assistance of Claude (Anthropic). Prompts used included filling missing horsepower values with cylinder-group averages, building the Keras regression DNN, and generating predictions for custom vehicle data. All code has been reviewed and understood by the author.

---

## Author

**Abdu Oumasse**  
MSc Business Analytics & AI — Ontario Tech University  
[LinkedIn](https://linkedin.com/in/abdu-oumasse) · [Portfolio](https://abduoumassevisualizationportfolio.netlify.app)
