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
teamform_scaling_factor = 0.5

driver_form_2025 = {
    'Max Verstappen': 0.96, 'Charles Leclerc': 0.99, 'George Russell': 0.98,
    'Lando Norris': 0.98, 'Oscar Piastri': 0.97, 'Kimi Andrea Antonelli': 0.99,
    'Lewis Hamilton': 1.00, 'Carlos Sainz': 1.12, 'Fernando Alonso': 1.00,
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

# Tire strategy mapping for different circuits
tire_strategy_impact = {
    'Melbourne': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Shanghai': {'soft_advantage': 0.2, 'medium_advantage': 0.1, 'hard_advantage': -0.1},
    'Suzuka': {'soft_advantage': 0.4, 'medium_advantage': 0.0, 'hard_advantage': -0.3},
    'Bahrain': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Jeddah': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Miami': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Imola': {'soft_advantage': 0.4, 'medium_advantage': 0.0, 'hard_advantage': -0.3},
    'Monaco': {'soft_advantage': 0.5, 'medium_advantage': 0.1, 'hard_advantage': -0.4},
    'Barcelona': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.1},
    'Montreal': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Spielberg': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Silverstone': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.1},
    'Spa': {'soft_advantage': 0.1, 'medium_advantage': 0.0, 'hard_advantage': 0.1},
    'Budapest': {'soft_advantage': 0.4, 'medium_advantage': 0.0, 'hard_advantage': -0.3},
    'Zandvoort': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Monza': {'soft_advantage': 0.1, 'medium_advantage': 0.0, 'hard_advantage': 0.1},
    'Baku': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Singapore': {'soft_advantage': 0.4, 'medium_advantage': 0.0, 'hard_advantage': -0.3},
    'Austin': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.1},
    'Mexico City': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Sao Paulo': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.1},
    'Las Vegas': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
    'Lusail': {'soft_advantage': 0.2, 'medium_advantage': 0.0, 'hard_advantage': -0.1},
    'Yas Marina': {'soft_advantage': 0.3, 'medium_advantage': 0.0, 'hard_advantage': -0.2},
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
        except Exception:
            continue
        for _, event in schedule.iterrows():
            location = event['Location']
            # Process Sprint Qualifying, Sprint Race, Qualifying, and Race
            for session_type in ['SQ', 'S', 'Q', 'R']:
                try:
                    session = fastf1.get_session(year, location, session_type)
                    session.load()
                    results = session.results
                    
                    # Get tire compounds used
                    try:
                        laps = session.laps
                        tire_data = laps.groupby('Driver')['Compound'].first().to_dict()
                    except:
                        tire_data = {}
                    
                    for _, row in results.iterrows():
                        # Get tire strategy
                        driver_code = row.get('Abbreviation', '')
                        tire_compound = tire_data.get(driver_code, 'MEDIUM')
                        
                        data.append({
                            'Driver': row['FullName'],
                            'Team': row['TeamName'],
                            'Circuit': location,
                            'Year': year,
                            'Session': session_type,
                            'GridPosition': row.get('GridPosition', np.nan),
                            'Position': row['Position'],
                            'TireCompound': tire_compound,
                            'IsSprint': session_type in ['SQ', 'S']
                        })
                except Exception:
                    continue
    return pd.DataFrame(data)

def calculate_qualifying_race_correlation(df):
    """Calculate correlation between qualifying and race positions"""
    race_data = df[df['Session'] == 'R'].copy()
    qual_data = df[df['Session'] == 'Q'].copy()
    
    # Merge qualifying and race data
    merged = pd.merge(
        qual_data[['Driver', 'Circuit', 'Year', 'Position']],
        race_data[['Driver', 'Circuit', 'Year', 'Position', 'GridPosition']],
        on=['Driver', 'Circuit', 'Year'],
        suffixes=('_qual', '_race')
    )
    
    correlation_dict = {}
    for circuit in merged['Circuit'].unique():
        circuit_data = merged[merged['Circuit'] == circuit]
        if len(circuit_data) > 5:  # Need sufficient data
            corr = circuit_data['Position_qual'].corr(circuit_data['Position_race'])
            correlation_dict[circuit] = corr if not np.isnan(corr) else 0.5
        else:
            correlation_dict[circuit] = 0.5  # Default correlation
    
    return correlation_dict

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

def get_tire_advantage(circuit, tire_compound):
    """Get tire advantage for given circuit and compound"""
    if circuit not in tire_strategy_impact:
        return 0.0
    
    compound_map = {
        'SOFT': 'soft_advantage',
        'MEDIUM': 'medium_advantage', 
        'HARD': 'hard_advantage'
    }
    
    advantage_key = compound_map.get(tire_compound, 'medium_advantage')
    return tire_strategy_impact[circuit].get(advantage_key, 0.0)

def display_model_performance(model, X, y):
    """Display model performance metrics by session type"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    # Get predictions for the entire dataset
    y_pred_all = model.predict(X)
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_r2 = r2_score(y, y_pred_all)
    
    print(f"Overall Performance:")
    print(f"  MAE: {overall_mae:.2f} positions")
    print(f"  R²:  {overall_r2:.3f}")
    print()
    
    # Performance by session type
    print("Performance by Session Type:")
    print("-" * 40)
    for session in ['Q', 'R', 'SQ', 'S']:
        mask = X['Session'] == session
        if mask.sum() > 0:
            y_true = y[mask]
            y_pred = model.predict(X[mask])
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            session_names = {
                'Q': 'Qualifying',
                'R': 'Race',
                'SQ': 'Sprint Qualifying',
                'S': 'Sprint Race'
            }
            print(f"  {session_names[session]:<18}: MAE = {mae:.2f}, R² = {r2:.3f}")
    
    print("="*60)

# Load or build dataset
if os.path.exists('f1_dataset_enhanced_2021_2024.csv'):
    print("Loading cached dataset...")
    df = pd.read_csv('f1_dataset_enhanced_2021_2024.csv')
else:
    print("Building enhanced dataset from FastF1...")
    df = build_dataset()
    df.to_csv('f1_dataset_enhanced_2021_2024.csv', index=False)

# Calculate qualifying-race correlation
if os.path.exists('qual_race_correlation.pkl'):
    print("Loading cached qualifying-race correlation...")
    with open('qual_race_correlation.pkl', 'rb') as f:
        qual_race_corr = pickle.load(f)
else:
    print("Calculating qualifying-race correlation...")
    qual_race_corr = calculate_qualifying_race_correlation(df)
    with open('qual_race_correlation.pkl', 'wb') as f:
        pickle.dump(qual_race_corr, f)

# Add enhanced features
df['DriverForm'] = df['Driver'].map(driver_form_2025).fillna(1.00) * form_influence
df['TireAdvantage'] = df.apply(lambda row: get_tire_advantage(row['Circuit'], row.get('TireCompound', 'MEDIUM')), axis=1)
df = df.dropna(subset=['Position'])

# Load or build DNF model
if os.path.exists('dnf_model_enhanced.pkl'):
    print("Loading cached DNF model...")
    with open('dnf_model_enhanced.pkl', 'rb') as f:
        dnf_model = pickle.load(f)
else:
    print("Building DNF model...")
    dnf_model = build_dnf_model()
    with open('dnf_model_enhanced.pkl', 'wb') as f:
        pickle.dump(dnf_model, f)

# Enhanced feature list
feature_cols = ['Driver', 'Circuit', 'Year', 'Session', 'DriverForm', 'TireCompound', 'TireAdvantage', 'IsSprint']
target = 'Position'
X = df[feature_cols]
y = df[target]

categorical = ['Driver', 'Circuit', 'Session', 'TireCompound']
numerical = ['Year', 'DriverForm', 'TireAdvantage', 'IsSprint']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numerical)
])

model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15))
])

# Train or load model
if os.path.exists('f1_model_enhanced_2025.pkl'):
    print("Loading pre-trained enhanced model...")
    model = joblib.load('f1_model_enhanced_2025.pkl')
else:
    print("Training enhanced model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'f1_model_enhanced_2025.pkl')

# ALWAYS display model performance metrics
display_model_performance(model, X, y)

# Enhanced prediction function
def predict_2025_session(circuit, session_type, is_sprint_weekend=False):
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
        'Yuki Tsunoda': 'RB', 'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Haas F1 Team'
    }

    # Determine optimal tire strategy for circuit
    tire_strategies = ['SOFT', 'MEDIUM', 'HARD']
    tire_advantages = {tire: get_tire_advantage(circuit, tire) for tire in tire_strategies}
    optimal_tire = max(tire_advantages, key=tire_advantages.get)

    prediction_input = []
    for driver in drivers_2025:
        team = teams_2025[driver]
        
        # Assign tire strategy (top teams get optimal, others vary)
        if team in ['Red Bull Racing', 'Ferrari', 'McLaren', 'Mercedes']:
            tire_compound = optimal_tire
        else:
            # Other teams might use different strategies
            tire_compound = np.random.choice(tire_strategies, p=[0.4, 0.4, 0.2])
        
        prediction_input.append({
            'Driver': driver,
            'Circuit': circuit,
            'Year': 2025,
            'Session': session_type,
            'DriverForm': driver_form_2025[driver] * form_influence,
            'Team': team,
            'TeamForm': team_form_2025[team] * form_influence,
            'TireCompound': tire_compound,
            'TireAdvantage': get_tire_advantage(circuit, tire_compound),
            'IsSprint': session_type in ['SQ', 'S']
        })

    df_pred = pd.DataFrame(prediction_input)
    df_pred['PredictedPosition'] = model.predict(df_pred[feature_cols])
    
    # Enhanced adjustments
    df_pred['AdjustedPrediction'] = (
        df_pred['PredictedPosition'] 
        - teamform_scaling_factor * (df_pred['TeamForm'] - 10)
        - df_pred['TireAdvantage'] * 2  # Tire strategy impact
    )
    
    df_pred['DNFChance'] = df_pred.apply(
        lambda row: dnf_model.get((row['Driver'], row['Circuit']), 0.02), axis=1
    )

    # Apply qualifying-race correlation for race predictions
    if session_type == 'R' and circuit in qual_race_corr:
        correlation_factor = qual_race_corr[circuit]
        # Generate mock qualifying positions (sorted by adjusted prediction)
        df_pred_sorted = df_pred.sort_values('AdjustedPrediction').reset_index(drop=True)
        df_pred_sorted['MockQualifyingPos'] = range(1, len(df_pred_sorted) + 1)
        
        # Apply grid position impact on race result
        df_pred_sorted['GridImpact'] = df_pred_sorted['MockQualifyingPos'] * correlation_factor * 0.3
        df_pred_sorted['FinalPosition'] = (
            df_pred_sorted['AdjustedPrediction'] 
            + df_pred_sorted['GridImpact']
            + df_pred_sorted['DNFChance'] * 12
        )
        df_pred = df_pred_sorted.sort_values('FinalPosition')
    elif session_type == 'R':
        df_pred['FinalPosition'] = df_pred['AdjustedPrediction'] + df_pred['DNFChance'] * 12
        df_pred = df_pred.sort_values('FinalPosition')
    else:
        df_pred = df_pred.sort_values('AdjustedPrediction')

    # Display results
    session_name = {
        'Q': 'Qualifying',
        'R': 'Race',
        'SQ': 'Sprint Qualifying', 
        'S': 'Sprint Race'
    }.get(session_type, session_type)
    
    print(f"\n=== Predicted Results for {session_name} at {circuit} ===")
    for i, row in enumerate(df_pred.itertuples(), 1):
        if session_type in ['R', 'S']:
            dnf_display = f"DNF: {row.DNFChance:.0%}" if row.DNFChance > 0.05 else ""
            tire_display = f"({row.TireCompound})"
            print(f"{i:2d}. {row.Driver:<20} {row.Team:<15} {tire_display:<8} - Pred: {row.PredictedPosition:.1f} | Adj: {row.AdjustedPrediction:.1f} {dnf_display}")
        else:
            tire_display = f"({row.TireCompound})"
            print(f"{i:2d}. {row.Driver:<20} {row.Team:<15} {tire_display:<8} - Predicted: {row.AdjustedPrediction:.1f}")

# Enhanced prediction execution
print("\nSelect a Grand Prix to simulate:")
for idx, (name, loc, is_sprint) in enumerate(calendar_2025, 1):
    sprint_indicator = " (Sprint Weekend)" if is_sprint else ""
    print(f"{idx:2d}. {name}{sprint_indicator}")

choice = int(input("Enter your choice: ")) - 1
selected_gp, circuit_code, is_sprint_weekend = calendar_2025[choice]
  
print(f"\n{'='*60}")
print(f"SIMULATING: {selected_gp}")
print(f"{'='*60}")

if is_sprint_weekend:
    print("\n🏁 SPRINT WEEKEND FORMAT")
    predict_2025_session(circuit_code, 'SQ', is_sprint_weekend)
    predict_2025_session(circuit_code, 'S', is_sprint_weekend)
    predict_2025_session(circuit_code, 'Q', is_sprint_weekend)
    predict_2025_session(circuit_code, 'R', is_sprint_weekend)
else:
    print("\n🏁 STANDARD WEEKEND FORMAT")
    predict_2025_session(circuit_code, 'Q', is_sprint_weekend)
    predict_2025_session(circuit_code, 'R', is_sprint_weekend)