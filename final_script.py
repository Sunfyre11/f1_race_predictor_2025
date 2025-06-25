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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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

def build_dataset_with_tire_analysis(start_year=2021, end_year=2024):
    """Enhanced dataset building with detailed tire compound analysis"""
    data = []
    tire_performance_data = []
    
    print("Building enhanced dataset with tire analysis...")
    
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"   Error getting schedule for {year}: {e}")
            continue
            
        for _, event in schedule.iterrows():
            location = event['Location']
            print(f"   Processing {location}...")
            
            # Process Sprint Qualifying, Sprint Race, Qualifying, and Race
            for session_type in ['SQ', 'S', 'Q', 'R']:
                try:
                    session = fastf1.get_session(year, location, session_type)
                    session.load()
                    results = session.results
                    
                    # Get detailed tire data
                    try:
                        laps = session.laps
                        if len(laps) > 0:
                            # Get tire compound usage and performance
                            for driver_code in laps['Driver'].unique():
                                driver_laps = laps[laps['Driver'] == driver_code]
                                if len(driver_laps) > 0:
                                    # Get most used compound
                                    compound_counts = driver_laps['Compound'].value_counts()
                                    if len(compound_counts) > 0:
                                        primary_compound = compound_counts.index[0]
                                        
                                        # Get average lap time for this compound
                                        compound_laps = driver_laps[driver_laps['Compound'] == primary_compound]
                                        valid_times = compound_laps['LapTime'].dropna()
                                        
                                        if len(valid_times) > 0:
                                            avg_lap_time = valid_times.mean().total_seconds()
                                            
                                            # Store tire performance data
                                            tire_performance_data.append({
                                                'Year': year,
                                                'Circuit': location,
                                                'Session': session_type,
                                                'Driver': driver_code,
                                                'Compound': primary_compound,
                                                'AvgLapTime': avg_lap_time,
                                                'LapCount': len(compound_laps)
                                            })
                    except Exception as e:
                        print(f"     Error processing tire data for {location} {session_type}: {e}")
                    
                    # Get basic tire compound data for results
                    try:
                        laps = session.laps
                        tire_data = laps.groupby('Driver')['Compound'].first().to_dict()
                    except:
                        tire_data = {}
                    
                    for _, row in results.iterrows():
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
                except Exception as e:
                    print(f"     Error processing {location} {session_type}: {e}")
                    continue
    
    return pd.DataFrame(data), pd.DataFrame(tire_performance_data)

def analyze_tire_performance_from_data(tire_performance_df, results_df):
    """Analyze tire performance based on actual lap times and race results"""
    print("Analyzing tire performance from historical data...")
    
    tire_advantages = {}
    
    # Method 1: Analyze based on lap times
    print("   Calculating tire advantages based on lap times...")
    for circuit in tire_performance_df['Circuit'].unique():
        circuit_data = tire_performance_df[
            (tire_performance_df['Circuit'] == circuit) & 
            (tire_performance_df['Session'].isin(['Q', 'R']))  # Focus on quali and race
        ]
        
        if len(circuit_data) < 10:  # Need sufficient data
            continue
            
        # Calculate average lap time by compound
        compound_times = circuit_data.groupby('Compound')['AvgLapTime'].mean()
        
        if 'MEDIUM' in compound_times.index:
            baseline = compound_times['MEDIUM']
            
            tire_advantages[circuit] = {
                'soft_advantage': (baseline - compound_times.get('SOFT', baseline)) / baseline * 100 if 'SOFT' in compound_times.index else 0.0,
                'medium_advantage': 0.0,
                'hard_advantage': (baseline - compound_times.get('HARD', baseline)) / baseline * 100 if 'HARD' in compound_times.index else 0.0
            }
        elif len(compound_times) >= 2:
            # Use fastest compound as baseline if no medium data
            baseline = compound_times.min()
            tire_advantages[circuit] = {}
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                if compound in compound_times.index:
                    advantage = (baseline - compound_times[compound]) / baseline * 100
                    tire_advantages[circuit][f'{compound.lower()}_advantage'] = advantage
                else:
                    tire_advantages[circuit][f'{compound.lower()}_advantage'] = 0.0
    
    # Method 2: Analyze based on race positions
    print("   Calculating tire advantages based on race positions...")
    race_data = results_df[results_df['Session'] == 'R']
    
    for circuit in race_data['Circuit'].unique():
        if circuit in tire_advantages:  # Already have lap time data
            continue
            
        circuit_data = race_data[race_data['Circuit'] == circuit]
        
        # Calculate average finishing position by tire compound
        position_by_compound = circuit_data.groupby('TireCompound')['Position'].mean()
        
        if len(position_by_compound) >= 2:
            if 'MEDIUM' in position_by_compound.index:
                baseline_pos = position_by_compound['MEDIUM']
                
                tire_advantages[circuit] = {
                    'soft_advantage': (baseline_pos - position_by_compound.get('SOFT', baseline_pos)) * 0.1,
                    'medium_advantage': 0.0,
                    'hard_advantage': (baseline_pos - position_by_compound.get('HARD', baseline_pos)) * 0.1
                }
            else:
                # Use best performing compound as baseline
                best_pos = position_by_compound.min()
                tire_advantages[circuit] = {}
                for compound in ['SOFT', 'MEDIUM', 'HARD']:
                    if compound in position_by_compound.index:
                        advantage = (best_pos - position_by_compound[compound]) * 0.1
                        tire_advantages[circuit][f'{compound.lower()}_advantage'] = advantage
                    else:
                        tire_advantages[circuit][f'{compound.lower()}_advantage'] = 0.0
    
    return tire_advantages

def get_tire_advantage_data_driven(circuit, tire_compound, tire_advantages_dict):
    """Get tire advantage based on historical data analysis"""
    if circuit not in tire_advantages_dict:
        return 0.0
    
    compound_map = {
        'SOFT': 'soft_advantage',
        'MEDIUM': 'medium_advantage', 
        'HARD': 'hard_advantage'
    }
    
    advantage_key = compound_map.get(tire_compound, 'medium_advantage')
    return tire_advantages_dict[circuit].get(advantage_key, 0.0)

def display_tire_advantage_comparison(tire_advantages_dict):
    """Display the calculated tire advantages"""
    print("\n" + "="*80)
    print("DATA-DRIVEN TIRE ADVANTAGES")
    print("="*80)
    
    print(f"{'Circuit':<15} {'Soft':<8} {'Medium':<8} {'Hard':<8} {'Optimal':<10}")
    print("-" * 60)
    
    for circuit, advantages in tire_advantages_dict.items():
        soft_adv = advantages.get('soft_advantage', 0.0)
        medium_adv = advantages.get('medium_advantage', 0.0)
        hard_adv = advantages.get('hard_advantage', 0.0)
        
        # Find optimal compound
        compound_advantages = {
            'SOFT': soft_adv,
            'MEDIUM': medium_adv,
            'HARD': hard_adv
        }
        optimal = max(compound_advantages, key=compound_advantages.get)
        
        print(f"{circuit:<15} {soft_adv:+6.2f}   {medium_adv:+6.2f}   {hard_adv:+6.2f}   {optimal:<10}")
    
    print("="*80)

def calculate_qualifying_race_correlation(df):
    """Calculate correlation between qualifying and race positions"""
    race_data = df[df['Session'] == 'R'].copy()
    qual_data = df[df['Session'] == 'Q'].copy()
    
    merged = pd.merge(
        qual_data[['Driver', 'Circuit', 'Year', 'Position']],
        race_data[['Driver', 'Circuit', 'Year', 'Position', 'GridPosition']],
        on=['Driver', 'Circuit', 'Year'],
        suffixes=('_qual', '_race')
    )
    
    correlation_dict = {}
    for circuit in merged['Circuit'].unique():
        circuit_data = merged[merged['Circuit'] == circuit]
        if len(circuit_data) > 5:
            corr = circuit_data['Position_qual'].corr(circuit_data['Position_race'])
            correlation_dict[circuit] = corr if not np.isnan(corr) else 0.5
        else:
            correlation_dict[circuit] = 0.5
    
    return correlation_dict

def build_dnf_model(start_year=2021, end_year=2024):
    """Build DNF probability model"""
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

def display_model_performance(model, X, y):
    """Display model performance metrics by session type"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    
    y_pred_all = model.predict(X)
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_r2 = r2_score(y, y_pred_all)
    
    print(f"Overall Performance:")
    print(f"   MAE: {overall_mae:.2f} positions")
    print(f"   R²:  {overall_r2:.3f}")
    print()
    
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
            print(f"   {session_names[session]:<18}: MAE = {mae:.2f}, R² = {r2:.3f}")
    
    print("="*60)

# Load or build dataset with tire analysis
if os.path.exists('f1_dataset_with_tires_2021_2024.csv') and os.path.exists('tire_performance_data_2021_2024.csv'):
    print("Loading cached datasets...")
    df = pd.read_csv('f1_dataset_with_tires_2021_2024.csv')
    tire_performance_df = pd.read_csv('tire_performance_data_2021_2024.csv')
else:
    print("Building enhanced datasets from FastF1...")
    df, tire_performance_df = build_dataset_with_tire_analysis()
    df.to_csv('f1_dataset_with_tires_2021_2024.csv', index=False)
    tire_performance_df.to_csv('tire_performance_data_2021_2024.csv', index=False)

# Calculate data-driven tire advantages
if os.path.exists('data_driven_tire_advantages.pkl'):
    print("Loading cached tire advantages...")
    with open('data_driven_tire_advantages.pkl', 'rb') as f:
        tire_advantages_dict = pickle.load(f)
else:
    print("Calculating data-driven tire advantages...")
    tire_advantages_dict = analyze_tire_performance_from_data(tire_performance_df, df)
    with open('data_driven_tire_advantages.pkl', 'wb') as f:
        pickle.dump(tire_advantages_dict, f)

# Display tire advantages
display_tire_advantage_comparison(tire_advantages_dict)

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

# Add enhanced features with data-driven tire advantages
df['DriverForm'] = df['Driver'].map(driver_form_2025).fillna(1.00) * form_influence
df['TireAdvantage'] = df.apply(
    lambda row: get_tire_advantage_data_driven(
        row['Circuit'], 
        row.get('TireCompound', 'MEDIUM'), 
        tire_advantages_dict
    ), axis=1
)
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
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20, min_samples_split=5))
])

# Train or load model
model_filename = 'f1_model_data_driven_tires_2025.pkl'
if os.path.exists(model_filename):
    print("Loading pre-trained data-driven model...")
    model = joblib.load(model_filename)
else:
    print("Training data-driven model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)

# Display model performance
display_model_performance(model, X, y)

# Enhanced prediction function with data-driven tire strategy
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

    # Determine optimal tire strategy based on data-driven analysis
    if circuit in tire_advantages_dict:
        tire_strategies = ['SOFT', 'MEDIUM', 'HARD']
        tire_advantages = {
            'SOFT': tire_advantages_dict[circuit].get('soft_advantage', 0.0),
            'MEDIUM': tire_advantages_dict[circuit].get('medium_advantage', 0.0),
            'HARD': tire_advantages_dict[circuit].get('hard_advantage', 0.0)
        }
        optimal_tire = max(tire_advantages, key=tire_advantages.get)
        
        print(f"Data-driven tire analysis for {circuit}:")
        for compound, advantage in tire_advantages.items():
            marker = " ← OPTIMAL" if compound == optimal_tire else ""
            print(f"   {compound}: {advantage:+.2f}{marker}")
    else:
        # Fallback if no data available
        optimal_tire = 'SOFT'
        print(f"No tire data available for {circuit}, defaulting to SOFT")

    prediction_input = []
    for driver in drivers_2025:
        team = teams_2025[driver]
        
        # Assign tire strategy based on data-driven analysis
        if team in ['Red Bull Racing', 'Ferrari', 'McLaren', 'Mercedes']:
            tire_compound = optimal_tire
        else:
            # Other teams might use different strategies
            tire_strategies = ['SOFT', 'MEDIUM', 'HARD']
            weights = [0.5, 0.3, 0.2] if optimal_tire == 'SOFT' else [0.3, 0.4, 0.3]
            tire_compound = np.random.choice(tire_strategies, p=weights)
        
        prediction_input.append({
            'Driver': driver,
            'Circuit': circuit,
            'Year': 2025,
            'Session': session_type,
            'DriverForm': driver_form_2025[driver] * form_influence,
            'Team': team,
            'TeamForm': team_form_2025[team] * form_influence,
            'TireCompound': tire_compound,
            'TireAdvantage': get_tire_advantage_data_driven(circuit, tire_compound, tire_advantages_dict),
            'IsSprint': session_type in ['SQ', 'S']
        })

    df_pred = pd.DataFrame(prediction_input)
    df_pred['PredictedPosition'] = model.predict(df_pred[feature_cols])
    
    # Enhanced adjustments with data-driven tire advantages
    df_pred['AdjustedPrediction'] = (
        df_pred['PredictedPosition'] 
        - teamform_scaling_factor * (df_pred['TeamForm'] - 10)
        - df_pred['TireAdvantage'] * 3  # Increased multiplier for data-driven advantages
    )
    
    df_pred['DNFChance'] = df_pred.apply(
        lambda row: dnf_model.get((row['Driver'], row['Circuit']), 0.02), axis=1
    )

    # Apply qualifying-race correlation for race predictions
    if session_type == 'R' and circuit in qual_race_corr:
        correlation_factor = qual_race_corr[circuit]
        df_pred_sorted = df_pred.sort_values('AdjustedPrediction').reset_index(drop=True)
        df_pred_sorted['MockQualifyingPos'] = range(1, len(df_pred_sorted) + 1)
        
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
            tire_adv_display = f"TA: {row.TireAdvantage:+.2f}"
            print(f"{i:2d}. {row.Driver:<20} {row.Team:<15} {tire_display:<8} {tire_adv_display:<9} - Pred: {row.PredictedPosition:.1f} | Adj: {row.AdjustedPrediction:.1f} {dnf_display}")
        else:
            tire_display = f"({row.TireCompound})"
            tire_adv_display = f"TA: {row.TireAdvantage:+.2f}"
            print(f"{i:2d}. {row.Driver:<20} {row.Team:<15} {tire_display:<8} {tire_adv_display:<9} - Predicted: {row.AdjustedPrediction:.1f}")

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