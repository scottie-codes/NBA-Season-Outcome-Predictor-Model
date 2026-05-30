# NBA Season Outcome Predictor and Team Analytics Dashboard

**WGU Computer Science Capstone: C964**

This program is a dashboard app that analyzes NBA team statistics and predicts season outcomes using machine learning. 
The models use the first 20 games of a season to make end of season win percentage and playoff predictions.
I built this project because I'm an enthusiastic NBA stats nerd and I thought it would be fun 
to try to use real data science models to predict useful outcomes of something I watch every day. Enjoy!

---

## Quick Start

### 1. Install Python
Requires Python 3.10 or higher (https://www.python.org/downloads/)

### 2. Set up the virtual environment: 
 ```bash
   python -m venv .venv
```

### 3. Activate virtual environment, then wait for it to finish: 
```bash
.venv\Scripts\activate
```



### 4. Install Dependencies and wait for it to finish.
```bash
pip install -r requirements.txt
```

### 5. Run the Application!
```bash
python src/app.py
```

---

## Project Structure
```
CAPSTONE/
	data/
		processed/              # Trained models and processed data
		raw/                      # Source data from nba_api, different endpoints
	src/
		app.py                  # Main app entry point
		data_collection.py      # Collected the data from nba_api
		make_master_features.py  # Feature engineering
		train_models.py            # Model training
		models/                 # Prediction and data loading
		ui/                      # User interface screens
	requirements.txt                # Python dependencies
	README.md	                # This document
```

---

## Features

### Prediction Screen
- Select any team and season (2015-2025)
- Predicts win % using Random Forest regression
- Predicts top half or bottom half finish using Random Forest classification (for playoff predicitons)
- Compares predicted vs. actual results
- Also gives model performance metrics in the right window

### Analysis Screen
- Visualizes NBA trends from 2015-2025
- Shows correlations between certain key stats and winning
- Highlights field goal efficiency, assists, turnovers

---

## Data Pipeline

1. **Data Collection**: `data_collection.py`
   - Pulls team stats from NBA API
   - Saves to `data/raw/`

2. **Feature Engineering**: `make_master_features.py`
   - Processes raw data into model features
   - Uses first 20 games of each season
   - Saves to `data/processed/master_features.csv`

3. **Model Training**: `train_models.py`
   - Trains Random Forest regressor and classifier
   - Saves models to `data/processed/`

---

## Models

| Model | Purpose                          | Validation Metrics |
|-------|----------------------------------|--------------------|
| Random Forest Regressor | Predict win percentage           | MAE, R²            |
| Random Forest Classifier | Predict top-half vs. bottom-half | ACC, AUC           |

---

## Technologies

- **Python 3.10+**
- **pandas** - Data Manipulation
- **scikit-learn** - Machine learning
- **matplotlib** - Visualization
- **tkinter** - UI/ Dashboard App
- **nba_api** - NBA statistics

---

## Author

Scott Curtis  
Western Governors University  
Computer Science Capstone: C964  
Jan 2026