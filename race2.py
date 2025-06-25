import fastf1
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

fastf1.Cache.enable_cache('f1_cache')
form_influence = 10.0

driver_form_2025 = {
    'Max Verstappen': 0.96, 'Charles Leclerc': 0.99, 'George Russell': 0.98,
    'Lando Norris': 0.98, 'Oscar Piastri': 0.97, 'Kimi Andrea Antonelli': 1.00,
    'Lewis Hamilton': 1.00, 'Carlos Sainz': 1.03, 'Fernando Alonso': 1.00,
    'Lance Stroll': 1.03, 'Alex Albon': 1.01, 'Isack Hadjar': 1.01,
    'Yuki Tsunoda': 1.02, 'Oliver Bearman': 1.02, 'Gabriel Bortoleto': 1.04,
    'Jack Doohan': 1.03, 'Nico Hulkenberg': 1.02, 'Liam Lawson': 1.03,
    'Pierre Gasly': 1.01, 'Esteban Ocon': 1.02,
}

team_form_2025 = {
    'Red Bull Racing': 1.00, 'Ferrari': 1.00, 'Mercedes': 0.98,
    'McLaren': 0.95, 'Aston Martin': 1.02, 'Williams': 1.03,
    'RB': 1.03, 'Haas F1 Team': 1.04, 'Kick Sauber': 1.04, 'Alpine': 1.05,
}

calendar_2025 = [
    ('Australian GP', 'Melbourne', False), ('Chinese GP', 'Shanghai', True),
    ('Japanese GP', 'Suzuka', False), ('Bahrain GP', 'Bahrain', False),
    ('Saudi Arabian GP', 'Jeddah', False), ('Miami GP', 'Miami', True),
    ('Emilia-Romagna GP', 'Imola', False), ('Monaco GP', 'Monaco', False),
    ('Spanish GP', 'Barcelona', False), ('Canadian GP', 'Montreal', False),
    ('Austrian GP', 'Spielberg', False), ('British GP', 'Silverstone', False),
    ('Belgian GP', 'Spa', True), ('Hungarian GP', 'Budapest', False),
    ('Dutch GP', 'Zandvoort', False), ('Italian GP', 'Monza', False),
    ('Azerbaijan GP', 'Baku', False), ('Singapore GP', 'Singapore', False),
    ('United States GP', 'Austin', True), ('Mexican GP', 'Mexico City', False),
    ('Brazilian GP', 'Sao Paulo', True), ('Las Vegas GP', 'Las Vegas', False),
    ('Qatar GP', 'Lusail', True), ('Abu Dhabi GP', 'Yas Marina', False)
]

def build_dataset(start_year=2021, end_year=2024):
    data = []
    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"⚠️ Could not load schedule for {year}: {e}")
            continue
        for _, event in schedule.iterrows():
            location = event['Location']
            for session_type in ['Q', 'R']:
                try:
                    session = fastf1.get_session(year, location, session_type)
                    session.load()
                    results = session.results
                    for _, row in results.iterrows():
                        data.append({
                            'Driver': row['FullName'],
                            'Team': row['TeamName'],
                            'Circuit': location,
                            'Year': year,
                            'Session': session_type,
                            'GridPosition': row.get('GridPosition', np.nan),
                            'Position': row['Position'],
                        })
                except Exception:
                    continue
    return pd.DataFrame(data)

def build_dnf_model(start_year=2021, end_year=2024):
    dnf_counts = {}
    starts = {}
    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year)
            for _, event in schedule.iterrows():
                location = event['Location']
                try:
                    session = fastf1.get_session(year, location, 'R')
                    session.load()
                    results = session.results
                    for _, row in results.iterrows():
                        driver = row['FullName']
                        status = row['Status']
                        key = (driver, location)
                        if key not in starts:
                            starts[key] = 0
                            dnf_counts[key] = 0
                        starts[key] += 1
                        if 'Finished' not in status:
                            dnf_counts[key] += 1
                except Exception:
                    continue
        except Exception:
            continue

    return {key: dnf_counts[key] / starts[key] if starts[key] > 0 else 0 for key in starts}

# Load or build dataset
if os.path.exists('f1_dataset_2021_2024.csv'):
    print("Loading cached dataset...")
    df = pd.read_csv('f1_dataset_2021_2024.csv')
else:
    print("Building dataset from FastF1...")
    df = build_dataset()
    df.to_csv('f1_dataset_2021_2024.csv', index=False)

df['DriverForm'] = df['Driver'].map(driver_form_2025).fillna(1.00) * form_influence
df['TeamForm'] = df['Team'].map(team_form_2025).fillna(1.00) * form_influence
df = df.dropna(subset=['Position'])

# Load or build DNF model
if os.path.exists('dnf_model.pkl'):
    print("Loading cached DNF model...")
    with open('dnf_model.pkl', 'rb') as f:
        dnf_model = pickle.load(f)
else:
    print("Building DNF model...")
    dnf_model = build_dnf_model()
    with open('dnf_model.pkl', 'wb') as f:
        pickle.dump(dnf_model, f)

# Define features and model pipeline
features = ['Driver', 'Team', 'Circuit', 'Year', 'Session', 'DriverForm', 'TeamForm']
target = 'Position'
X = df[features]
y = df[target]

categorical = ['Driver', 'Team', 'Circuit', 'Session']
numerical = ['Year', 'DriverForm', 'TeamForm']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numerical)
])

model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train or load model
if os.path.exists('f1_model_2025.pkl'):
    print("Loading pre-trained model...")
    model = joblib.load('f1_model_2025.pkl')
else:
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'f1_model_2025.pkl')

    # Evaluate
    print("\n=== Model Accuracy by Session Type ===")
    for session in ['Q', 'R']:
        mask = X_test['Session'] == session
        if mask.sum() > 0:
            y_true = y_test[mask]
            y_pred = model.predict(X_test[mask])
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"{session}: MAE = {mae:.2f}, R² = {r2:.2f}")

# Prediction function
def predict_2025_session(circuit, session_type):
    drivers_2025 = list(driver_form_2025.keys())
    teams_2025 = {
        'Max Verstappen': 'Red Bull Racing', 'Charles Leclerc': 'Ferrari',
        'George Russell': 'Mercedes', 'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren', 'Kimi Andrea Antonelli': 'Mercedes',
        'Lewis Hamilton': 'Ferrari', 'Carlos Sainz': 'Williams',
        'Fernando Alonso': 'Aston Martin', 'Lance Stroll': 'Aston Martin',
        'Alex Albon': 'Williams', 'Isack Hadjar': 'RB', 'Liam Lawson': 'RB',
        'Oliver Bearman': 'Haas F1 Team', 'Gabriel Bortoleto': 'Kick Sauber',
        'Jack Doohan': 'Alpine', 'Nico Hulkenberg': 'Kick Sauber',
        'Yuki Tsunoda': 'Red Bull Racing', 'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Haas F1 Team'
    }

    prediction_input = []
    for driver in drivers_2025:
        team = teams_2025[driver]
        prediction_input.append({
            'Driver': driver,
            'Team': team,
            'Circuit': circuit,
            'Year': 2025,
            'Session': session_type,
            'DriverForm': driver_form_2025[driver] * form_influence,
            'TeamForm': team_form_2025[team] * form_influence
        })

    df_pred = pd.DataFrame(prediction_input)
    predicted_positions = model.predict(df_pred)
    df_pred['PredictedPosition'] = predicted_positions
    df_pred['DNFChance'] = df_pred.apply(
        lambda row: dnf_model.get((row['Driver'], row['Circuit']), 0.02), axis=1
    )

    if session_type == 'R':
        df_pred['AdjustedPosition'] = df_pred['PredictedPosition'] + df_pred['DNFChance'] * 10
        df_pred = df_pred.sort_values('AdjustedPosition')
    else:
        df_pred = df_pred.sort_values('PredictedPosition')

    print(f"\n=== Predicted Results for {session_type} at {circuit} ===")
    for i, row in enumerate(df_pred.itertuples(), 1):
        if session_type == 'R':
            print(f"{i}. {row.Driver} ({row.Team}) - Predicted: {row.PredictedPosition:.2f} | Adjusted: {row.AdjustedPosition:.2f} | DNF Chance: {row.DNFChance:.0%}")
        else:
            print(f"{i}. {row.Driver} ({row.Team}) - Predicted: {row.PredictedPosition:.2f}")

# Run prediction
print("\nSelect a Grand Prix to simulate:")
for idx, (name, loc, _) in enumerate(calendar_2025, 1):
    print(f"{idx}. {name}")

choice = int(input("Enter your choice: ")) - 1
selected_gp, circuit_code, _ = calendar_2025[choice]

predict_2025_session(circuit_code, 'Q')
predict_2025_session(circuit_code, 'R')
