import joblib
import pandas as pd
import os
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from admindashboard.models import SoilMoistureRecord, SoilMoisturePrediction
import json
import logging
import numpy as np
from datetime import datetime
import pandas as pd
import os
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import logging
import numpy as np
from datetime import datetime

# Define paths for model and metrics
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'soil_status_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'label_encoder.pkl')
METRICS_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'model_metrics.json')

# Initialize model variables
model = None
scaler = None
label_encoder = None
feature_selector = None
selected_features = None

# Define paths for model and metrics
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'soil_status_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'label_encoder.pkl')
METRICS_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'model_metrics.json')

# Initialize model variables
model = None
scaler = None
label_encoder = None
feature_selector = None
selected_features = None

logger = logging.getLogger(__name__)

def determine_soil_status(soil_moisture, humidity, temperature):
    """
    Determine soil status based on the three factors
    
    Args:
        soil_moisture: Soil moisture percentage
        humidity: Humidity percentage
        temperature: Temperature in Celsius
    
    Returns:
        str: Soil status ('Critical Low', 'Dry', 'Normal', 'Wet', 'Critical High')
    """
    # Initialize scores for each factor
    try:
        soil_moisture = float(soil_moisture)
        humidity = float(humidity)
        temperature = float(temperature)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting inputs to float: {e}")
        return "Unknown"
    
    # Soil Moisture scoring
    if soil_moisture < 30:
        moisture_score = -2  # Critical Low/Dry
    elif 30 <= soil_moisture <= 60:
        moisture_score = 0   # Normal
    else:  # > 60
        moisture_score = 2   # Critical High/Wet
    
    # Humidity scoring
    if humidity < 40:
        humidity_score = -2  # Critical Low/Dry
    elif 40 <= humidity <= 80:
        humidity_score = 0   # Normal
    else:  # > 80
        humidity_score = 2   # Critical High/Wet
    
    # Temperature scoring (inverted logic)
    if temperature > 35:
        temp_score = -2      # Critical Low/Dry
    elif 20 <= temperature <= 35:
        temp_score = 0       # Normal
    else:  # < 20
        temp_score = 2       # Critical High/Wet
    
    # Calculate total score
    total_score = moisture_score + humidity_score + temp_score
    
    # Determine status based on total score
    if total_score <= -4:
        return "Critical Low"
    elif total_score <= -1:
        return "Dry"
    elif total_score <= 1:
        return "Normal"
    elif total_score <= 4:
        return "Wet"
    else:
        return "Critical High"

def get_irrigation_recommendation(soil_status):
    """
    Get irrigation recommendation based on soil status
    
    Args:
        soil_status: Soil status string
    
    Returns:
        str: Irrigation recommendation
    """
    irrigation_map = {
        "Critical Low": "Intense Irrigation",
        "Dry": "Intense Irrigation",
        "Normal": "Reduced Irrigation",
        "Wet": "No Irrigation",
        "Critical High": "No Irrigation"
    }
    return irrigation_map.get(soil_status, "Normal Irrigation")

def create_features(df):
    """Create engineered features for better prediction"""
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(lambda x: 
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else
        'Autumn' if x in [9, 10, 11] else 'Winter'
    )
    
    # Cyclical time features for better temporal representation
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Extended lag features
    df['moisture_lag_1'] = df['soil_moisture_percent'].shift(1)
    df['moisture_lag_2'] = df['soil_moisture_percent'].shift(2)
    df['moisture_lag_3'] = df['soil_moisture_percent'].shift(3)
    df['temp_lag_1'] = df['temperature_celsius'].shift(1)
    df['temp_lag_2'] = df['temperature_celsius'].shift(2)
    df['humidity_lag_1'] = df['humidity_percent'].shift(1)
    df['humidity_lag_2'] = df['humidity_percent'].shift(2)
    
    # Multiple rolling windows for better trend capture
    df['moisture_rolling_3'] = df['soil_moisture_percent'].rolling(window=3, min_periods=1).mean()
    df['moisture_rolling_6'] = df['soil_moisture_percent'].rolling(window=6, min_periods=1).mean()
    df['moisture_rolling_12'] = df['soil_moisture_percent'].rolling(window=12, min_periods=1).mean()
    df['temp_rolling_3'] = df['temperature_celsius'].rolling(window=3, min_periods=1).mean()
    df['temp_rolling_6'] = df['temperature_celsius'].rolling(window=6, min_periods=1).mean()
    df['humidity_rolling_3'] = df['humidity_percent'].rolling(window=3, min_periods=1).mean()
    df['humidity_rolling_6'] = df['humidity_percent'].rolling(window=6, min_periods=1).mean()
    
    # Rolling standard deviations (volatility features)
    df['moisture_rolling_std'] = df['soil_moisture_percent'].rolling(window=6, min_periods=1).std()
    df['temp_rolling_std'] = df['temperature_celsius'].rolling(window=6, min_periods=1).std()
    df['humidity_rolling_std'] = df['humidity_percent'].rolling(window=6, min_periods=1).std()
    
    # Rate of change features
    df['moisture_change'] = df['soil_moisture_percent'].diff()
    df['moisture_change_2'] = df['soil_moisture_percent'].diff(2)
    df['temp_change'] = df['temperature_celsius'].diff()
    df['temp_change_2'] = df['temperature_celsius'].diff(2)
    df['humidity_change'] = df['humidity_percent'].diff()
    df['humidity_change_2'] = df['humidity_percent'].diff(2)
    
    # Advanced interaction features
    df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']
    df['moisture_temp_ratio'] = df['soil_moisture_percent'] / (df['temperature_celsius'] + 1)
    df['moisture_humidity_ratio'] = df['soil_moisture_percent'] / (df['humidity_percent'] + 1)
    df['temp_humidity_ratio'] = df['temperature_celsius'] / (df['humidity_percent'] + 1)
    
    # Polynomial features for non-linear relationships
    df['moisture_squared'] = df['soil_moisture_percent'] ** 2
    df['temp_squared'] = df['temperature_celsius'] ** 2
    df['humidity_squared'] = df['humidity_percent'] ** 2
    
    # Composite indices
    df['dryness_index'] = (df['temperature_celsius'] / (df['humidity_percent'] + 1)) / (df['soil_moisture_percent'] + 1)
    df['moisture_deficit'] = np.maximum(0, 60 - df['soil_moisture_percent'])  # Deficit from optimal
    df['heat_stress_index'] = np.maximum(0, df['temperature_celsius'] - 30)  # Heat above comfort
    df['humidity_stress_index'] = np.abs(df['humidity_percent'] - 60)  # Distance from optimal humidity
    
    # Stability features (how stable are the conditions)
    df['moisture_stability'] = 1 / (df['moisture_rolling_std'] + 0.1)
    df['temp_stability'] = 1 / (df['temp_rolling_std'] + 0.1)
    df['humidity_stability'] = 1 / (df['humidity_rolling_std'] + 0.1)
    
    # Risk indicators
    df['critical_low_risk'] = ((df['soil_moisture_percent'] < 30) & 
                               (df['temperature_celsius'] > 30) & 
                               (df['humidity_percent'] < 50)).astype(int)
    df['critical_high_risk'] = ((df['soil_moisture_percent'] > 70) & 
                                (df['temperature_celsius'] < 25) & 
                                (df['humidity_percent'] > 75)).astype(int)
    
    return df

def train_model():
    global model, scaler, label_encoder
    try:
        # Fetch historical data
        records = SoilMoistureRecord.objects.all().values(
            'soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 'timestamp'
        )
        df = pd.DataFrame(records)
        
        # Check if there is enough data to train
        if len(df) < 50:
            logger.error(f"Insufficient data to train the model: {len(df)} records found. At least 50 records are recommended.")
            raise ValueError(f"Insufficient data to train the model. Found {len(df)} records, but at least 50 are recommended for meaningful predictions.")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create soil status labels based on the three factors
        df['status'] = df.apply(
            lambda row: determine_soil_status(
                row['soil_moisture_percent'],
                row['humidity_percent'],
                row['temperature_celsius']
            ), axis=1
        )
        
        # Create irrigation recommendations
        df['irrigation_recommendation'] = df['status'].apply(get_irrigation_recommendation)
        
        # Create engineered features
        df = create_features(df)
        
        # Define features for training - expanded feature set
        feature_columns = [
            # Core features
            'soil_moisture_percent', 'temperature_celsius', 'humidity_percent',
            
            # Cyclical time features
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 
            'month_sin', 'month_cos',
            
            # Extended lag features
            'moisture_lag_1', 'moisture_lag_2', 'moisture_lag_3',
            'temp_lag_1', 'temp_lag_2', 'humidity_lag_1', 'humidity_lag_2',
            
            # Multiple rolling windows
            'moisture_rolling_3', 'moisture_rolling_6', 'moisture_rolling_12',
            'temp_rolling_3', 'temp_rolling_6', 'humidity_rolling_3', 'humidity_rolling_6',
            
            # Volatility features
            'moisture_rolling_std', 'temp_rolling_std', 'humidity_rolling_std',
            
            # Rate of change features
            'moisture_change', 'moisture_change_2', 'temp_change', 'temp_change_2',
            'humidity_change', 'humidity_change_2',
            
            # Interaction and ratio features
            'temp_humidity_interaction', 'moisture_temp_ratio', 'moisture_humidity_ratio', 'temp_humidity_ratio',
            
            # Polynomial features
            'moisture_squared', 'temp_squared', 'humidity_squared',
            
            # Composite indices
            'dryness_index', 'moisture_deficit', 'heat_stress_index', 'humidity_stress_index',
            
            # Stability features
            'moisture_stability', 'temp_stability', 'humidity_stability',
            
            # Risk indicators
            'critical_low_risk', 'critical_high_risk'
        ]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['status'].copy()
        
        # Remove rows with NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 30:  # Reduced minimum since we have more features
            logger.error(f"After feature engineering, only {len(X)} valid samples remain. At least 30 are needed.")
            raise ValueError(f"After feature engineering, only {len(X)} valid samples remain. At least 30 are needed.")
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Feature selection using mutual information
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        # Encode target for feature selection
        temp_label_encoder = LabelEncoder()
        y_temp_encoded = temp_label_encoder.fit_transform(y)
        
        # Select top features
        k_best = min(25, len(feature_columns))  # Select top 25 features or all if less
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        X_selected = selector.fit_transform(X, y_temp_encoded)
        
        # Get selected feature names
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Ensemble approach with multiple models
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
        from sklearn.svm import SVC
        
        # Create multiple classifiers
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Create voting classifier
        model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get class names for reporting
        class_names = label_encoder.classes_
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Save model, scaler, label encoder, and feature selector
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        
        # Save feature selector and selected features
        SELECTOR_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_selector.pkl')
        SELECTED_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'selected_features.json')
        
        joblib.dump(selector, SELECTOR_PATH)
        with open(SELECTED_FEATURES_PATH, 'w') as f:
            json.dump(selected_features, f)
        
        logger.info(f"Model, scaler, label encoder, and feature selector saved successfully")
        
        # Save metrics with additional information
        metrics = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(class_report['weighted avg']['precision'] * 100, 2),
            'recall': round(class_report['weighted avg']['recall'] * 100, 2),
            'f1_score': round(class_report['weighted avg']['f1-score'] * 100, 2),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'selected_features_count': len(selected_features),
            'class_distribution': {cls: int(np.sum(y == cls)) for cls in class_names},
            'confusion_matrix': conf_matrix.tolist(),
            'selected_features': selected_features,
            'model_type': 'VotingClassifier (RF + GB + ET)',
            'class_wise_performance': {
                cls: {
                    'precision': round(class_report[cls]['precision'] * 100, 2),
                    'recall': round(class_report[cls]['recall'] * 100, 2),
                    'f1_score': round(class_report[cls]['f1-score'] * 100, 2)
                } for cls in class_names
            }
        }
        
        try:
            with open(METRICS_PATH, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved successfully: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        raise

# Load or train the model
try:
    SELECTOR_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_selector.pkl')
    SELECTED_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'selected_features.json')
    
    if (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and 
        os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(SELECTOR_PATH)):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        feature_selector = joblib.load(SELECTOR_PATH)
        
        # Load selected features
        if os.path.exists(SELECTED_FEATURES_PATH):
            with open(SELECTED_FEATURES_PATH, 'r') as f:
                selected_features = json.load(f)
        
        logger.info(f"Model, scaler, label encoder, and feature selector loaded successfully")
    else:
        logger.warning(f"Model files not found. Training a new model...")
        model, _ = train_model()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    model = None
    scaler = None
    label_encoder = None
    feature_selector = None
    selected_features = None

def predict_soil_status(soil_moisture, temperature, humidity, location=None):
    """
    Predict soil status and irrigation recommendation
    
    Args:
        soil_moisture: Current soil moisture percentage
        temperature: Current temperature in Celsius
        humidity: Current humidity percentage
        location: Optional location parameter
    
    Returns:
        dict: Contains soil_status, irrigation_recommendation, and confidence
    """
    try:
        # Rule-based prediction (primary method)
        rule_based_status = determine_soil_status(soil_moisture, humidity, temperature)
        rule_based_irrigation = get_irrigation_recommendation(rule_based_status)
        
        # If model is available, use it for enhanced prediction
        if (model is not None and scaler is not None and label_encoder is not None and 
            feature_selector is not None and selected_features is not None):
            now = datetime.now()
            
            # Create a comprehensive feature set
            input_data = pd.DataFrame({
                'soil_moisture_percent': [soil_moisture],
                'temperature_celsius': [temperature],
                'humidity_percent': [humidity],
                'hour_sin': [np.sin(2 * np.pi * now.hour / 24)],
                'hour_cos': [np.cos(2 * np.pi * now.hour / 24)],
                'day_of_year_sin': [np.sin(2 * np.pi * now.timetuple().tm_yday / 365)],
                'day_of_year_cos': [np.cos(2 * np.pi * now.timetuple().tm_yday / 365)],
                'month_sin': [np.sin(2 * np.pi * now.month / 12)],
                'month_cos': [np.cos(2 * np.pi * now.month / 12)],
                'moisture_lag_1': [soil_moisture],
                'moisture_lag_2': [soil_moisture],
                'moisture_lag_3': [soil_moisture],
                'temp_lag_1': [temperature],
                'temp_lag_2': [temperature],
                'humidity_lag_1': [humidity],
                'humidity_lag_2': [humidity],
                'moisture_rolling_3': [soil_moisture],
                'moisture_rolling_6': [soil_moisture],
                'moisture_rolling_12': [soil_moisture],
                'temp_rolling_3': [temperature],
                'temp_rolling_6': [temperature],
                'humidity_rolling_3': [humidity],
                'humidity_rolling_6': [humidity],
                'moisture_rolling_std': [0.1],  # Assume stable conditions
                'temp_rolling_std': [0.1],
                'humidity_rolling_std': [0.1],
                'moisture_change': [0.0],
                'moisture_change_2': [0.0],
                'temp_change': [0.0],
                'temp_change_2': [0.0],
                'humidity_change': [0.0],
                'humidity_change_2': [0.0],
                'temp_humidity_interaction': [temperature * humidity],
                'moisture_temp_ratio': [soil_moisture / (temperature + 1)],
                'moisture_humidity_ratio': [soil_moisture / (humidity + 1)],
                'temp_humidity_ratio': [temperature / (humidity + 1)],
                'moisture_squared': [soil_moisture ** 2],
                'temp_squared': [temperature ** 2],
                'humidity_squared': [humidity ** 2],
                'dryness_index': [(temperature / (humidity + 1)) / (soil_moisture + 1)],
                'moisture_deficit': [max(0, 60 - soil_moisture)],
                'heat_stress_index': [max(0, temperature - 30)],
                'humidity_stress_index': [abs(humidity - 60)],
                'moisture_stability': [10.0],  # Assume stable
                'temp_stability': [10.0],
                'humidity_stability': [10.0],
                'critical_low_risk': [float(soil_moisture < 30 and temperature > 30 and humidity < 50)],
                'critical_high_risk': [float(soil_moisture > 70 and temperature < 25 and humidity > 75)]
            })
            
            # Select only the features that were selected during training
            input_data_selected = input_data[selected_features]
            
            # Handle any missing columns
            for col in selected_features:
                if col not in input_data_selected.columns:
                    input_data_selected[col] = 0.0
            
            # Scale features
            input_scaled = scaler.transform(input_data_selected)
            
            # Make prediction
            prediction_encoded = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            ml_status = label_encoder.inverse_transform([prediction_encoded])[0]
            ml_irrigation = get_irrigation_recommendation(ml_status)
            
            # Get confidence (max probability)
            confidence = float(max(prediction_proba))
            
            # Use ML prediction if confidence is high, otherwise use rule-based
            if confidence > 0.7:  # Increased threshold for higher confidence
                final_status = ml_status
                final_irrigation = ml_irrigation
                method = "Enhanced ML Model"
            else:
                final_status = rule_based_status
                final_irrigation = rule_based_irrigation
                method = "Rule-based (Low ML Confidence)"
                confidence = 0.95  # High confidence for rule-based
            
        else:
            # Use rule-based prediction only
            final_status = rule_based_status
            final_irrigation = rule_based_irrigation
            method = "Rule-based"
            confidence = 0.95
        
        result = {
            'status': final_status,
            'irrigation_recommendation': final_irrigation,
            'confidence': round(confidence * 100, 1),
            'method': method,
            'input_values': {
                'soil_moisture': soil_moisture,
                'temperature': temperature,
                'humidity': humidity
            }
        }
        
        logger.info(f"Prediction made for {location or 'unknown location'}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise

def get_model_metrics():
    """Get model performance metrics"""
    try:
        logger.info(f"Loading metrics from {METRICS_PATH}")
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
                logger.info(f"Metrics loaded: {metrics}")
                return metrics
        logger.warning(f"Metrics file not found at {METRICS_PATH}")
        return {
            'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
            'training_samples': None, 'class_distribution': None
        }
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}", exc_info=True)
        return {
            'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
            'training_samples': None, 'class_distribution': None
        }

def get_irrigation_schedule_recommendation(soil_status, current_weather=None):
    """
    Get detailed irrigation schedule recommendation
    
    Args:
        soil_status: Current soil status
        current_weather: Optional weather information
    
    Returns:
        dict: Detailed irrigation schedule
    """
    schedules = {
        "Critical Low": {
            "urgency": "Immediate",
            "frequency": "Every 2-3 hours",
            "duration": "15-20 minutes per session",
            "water_amount": "Heavy watering",
            "monitoring": "Check every hour"
        },
        "Dry": {
            "urgency": "Within 2-4 hours",
            "frequency": "Every 4-6 hours",
            "duration": "10-15 minutes per session",
            "water_amount": "Moderate to heavy watering",
            "monitoring": "Check every 2-3 hours"
        },
        "Normal": {
            "urgency": "Within 6-12 hours",
            "frequency": "Every 12-24 hours",
            "duration": "5-10 minutes per session",
            "water_amount": "Light to moderate watering",
            "monitoring": "Check twice daily"
        },
        "Wet": {
            "urgency": "Not required",
            "frequency": "Monitor only",
            "duration": "No watering",
            "water_amount": "No watering",
            "monitoring": "Check daily"
        },
        "Critical High": {
            "urgency": "Stop all irrigation",
            "frequency": "No watering",
            "duration": "No watering",
            "water_amount": "No watering - risk of root rot",
            "monitoring": "Check for drainage issues"
        }
    }
    
    return schedules.get(soil_status, schedules["Normal"])

logger = logging.getLogger(__name__)

def determine_soil_status(soil_moisture, humidity, temperature):
    """
    Determine soil status based on the three factors
    
    Args:
        soil_moisture: Soil moisture percentage
        humidity: Humidity percentage
        temperature: Temperature in Celsius
    
    Returns:
        str: Soil status ('Critical Low', 'Dry', 'Normal', 'Wet', 'Critical High')
    """
    # Initialize scores for each factor
    try:
        soil_moisture = float(soil_moisture)
        humidity = float(humidity)
        temperature = float(temperature)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting inputs to float: {e}")
        return "Unknown"
    
    # Soil Moisture scoring
    if soil_moisture < 30:
        moisture_score = -2  # Critical Low/Dry
    elif 30 <= soil_moisture <= 60:
        moisture_score = 0   # Normal
    else:  # > 60
        moisture_score = 2   # Critical High/Wet
    
    # Humidity scoring
    if humidity < 40:
        humidity_score = -2  # Critical Low/Dry
    elif 40 <= humidity <= 80:
        humidity_score = 0   # Normal
    else:  # > 80
        humidity_score = 2   # Critical High/Wet
    
    # Temperature scoring (inverted logic)
    if temperature > 35:
        temp_score = -2      # Critical Low/Dry
    elif 20 <= temperature <= 35:
        temp_score = 0       # Normal
    else:  # < 20
        temp_score = 2       # Critical High/Wet
    
    # Calculate total score
    total_score = moisture_score + humidity_score + temp_score
    
    # Determine status based on total score
    if total_score <= -4:
        return "Critical Low"
    elif total_score <= -1:
        return "Dry"
    elif total_score <= 1:
        return "Normal"
    elif total_score <= 4:
        return "Wet"
    else:
        return "Critical High"

def get_irrigation_recommendation(soil_status):
    """
    Get irrigation recommendation based on soil status
    
    Args:
        soil_status: Soil status string
    
    Returns:
        str: Irrigation recommendation
    """
    irrigation_map = {
        "Critical Low": "Intense Irrigation",
        "Dry": "Intense Irrigation",
        "Normal": "Reduced Irrigation",
        "Wet": "No Irrigation",
        "Critical High": "No Irrigation"
    }
    return irrigation_map.get(soil_status, "Normal Irrigation")

def create_features(df):
    """Create engineered features for better prediction"""
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(lambda x: 
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else
        'Autumn' if x in [9, 10, 11] else 'Winter'
    )
    
    # Cyclical time features for better temporal representation
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Extended lag features
    df['moisture_lag_1'] = df['soil_moisture_percent'].shift(1)
    df['moisture_lag_2'] = df['soil_moisture_percent'].shift(2)
    df['moisture_lag_3'] = df['soil_moisture_percent'].shift(3)
    df['temp_lag_1'] = df['temperature_celsius'].shift(1)
    df['temp_lag_2'] = df['temperature_celsius'].shift(2)
    df['humidity_lag_1'] = df['humidity_percent'].shift(1)
    df['humidity_lag_2'] = df['humidity_percent'].shift(2)
    
    # Multiple rolling windows for better trend capture
    df['moisture_rolling_3'] = df['soil_moisture_percent'].rolling(window=3, min_periods=1).mean()
    df['moisture_rolling_6'] = df['soil_moisture_percent'].rolling(window=6, min_periods=1).mean()
    df['moisture_rolling_12'] = df['soil_moisture_percent'].rolling(window=12, min_periods=1).mean()
    df['temp_rolling_3'] = df['temperature_celsius'].rolling(window=3, min_periods=1).mean()
    df['temp_rolling_6'] = df['temperature_celsius'].rolling(window=6, min_periods=1).mean()
    df['humidity_rolling_3'] = df['humidity_percent'].rolling(window=3, min_periods=1).mean()
    df['humidity_rolling_6'] = df['humidity_percent'].rolling(window=6, min_periods=1).mean()
    
    # Rolling standard deviations (volatility features)
    df['moisture_rolling_std'] = df['soil_moisture_percent'].rolling(window=6, min_periods=1).std()
    df['temp_rolling_std'] = df['temperature_celsius'].rolling(window=6, min_periods=1).std()
    df['humidity_rolling_std'] = df['humidity_percent'].rolling(window=6, min_periods=1).std()
    
    # Rate of change features
    df['moisture_change'] = df['soil_moisture_percent'].diff()
    df['moisture_change_2'] = df['soil_moisture_percent'].diff(2)
    df['temp_change'] = df['temperature_celsius'].diff()
    df['temp_change_2'] = df['temperature_celsius'].diff(2)
    df['humidity_change'] = df['humidity_percent'].diff()
    df['humidity_change_2'] = df['humidity_percent'].diff(2)
    
    # Advanced interaction features
    df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']
    df['moisture_temp_ratio'] = df['soil_moisture_percent'] / (df['temperature_celsius'] + 1)
    df['moisture_humidity_ratio'] = df['soil_moisture_percent'] / (df['humidity_percent'] + 1)
    df['temp_humidity_ratio'] = df['temperature_celsius'] / (df['humidity_percent'] + 1)
    
    # Polynomial features for non-linear relationships
    df['moisture_squared'] = df['soil_moisture_percent'] ** 2
    df['temp_squared'] = df['temperature_celsius'] ** 2
    df['humidity_squared'] = df['humidity_percent'] ** 2
    
    # Composite indices
    df['dryness_index'] = (df['temperature_celsius'] / (df['humidity_percent'] + 1)) / (df['soil_moisture_percent'] + 1)
    df['moisture_deficit'] = np.maximum(0, 60 - df['soil_moisture_percent'])  # Deficit from optimal
    df['heat_stress_index'] = np.maximum(0, df['temperature_celsius'] - 30)  # Heat above comfort
    df['humidity_stress_index'] = np.abs(df['humidity_percent'] - 60)  # Distance from optimal humidity
    
    # Stability features (how stable are the conditions)
    df['moisture_stability'] = 1 / (df['moisture_rolling_std'] + 0.1)
    df['temp_stability'] = 1 / (df['temp_rolling_std'] + 0.1)
    df['humidity_stability'] = 1 / (df['humidity_rolling_std'] + 0.1)
    
    # Risk indicators
    df['critical_low_risk'] = ((df['soil_moisture_percent'] < 30) & 
                               (df['temperature_celsius'] > 30) & 
                               (df['humidity_percent'] < 50)).astype(int)
    df['critical_high_risk'] = ((df['soil_moisture_percent'] > 70) & 
                                (df['temperature_celsius'] < 25) & 
                                (df['humidity_percent'] > 75)).astype(int)
    
    return df

def train_model():
    global model, scaler, label_encoder
    try:
        # Fetch historical data
        records = SoilMoistureRecord.objects.all().values(
            'soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 'timestamp'
        )
        df = pd.DataFrame(records)
        
        # Check if there is enough data to train
        if len(df) < 50:
            logger.error(f"Insufficient data to train the model: {len(df)} records found. At least 50 records are recommended.")
            raise ValueError(f"Insufficient data to train the model. Found {len(df)} records, but at least 50 are recommended for meaningful predictions.")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create soil status labels based on the three factors
        df['status'] = df.apply(
            lambda row: determine_soil_status(
                row['soil_moisture_percent'],
                row['humidity_percent'],
                row['temperature_celsius']
            ), axis=1
        )
        
        # Create irrigation recommendations
        df['irrigation_recommendation'] = df['status'].apply(get_irrigation_recommendation)
        
        # Create engineered features
        df = create_features(df)
        
        # Define features for training - expanded feature set
        feature_columns = [
            # Core features
            'soil_moisture_percent', 'temperature_celsius', 'humidity_percent',
            
            # Cyclical time features
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 
            'month_sin', 'month_cos',
            
            # Extended lag features
            'moisture_lag_1', 'moisture_lag_2', 'moisture_lag_3',
            'temp_lag_1', 'temp_lag_2', 'humidity_lag_1', 'humidity_lag_2',
            
            # Multiple rolling windows
            'moisture_rolling_3', 'moisture_rolling_6', 'moisture_rolling_12',
            'temp_rolling_3', 'temp_rolling_6', 'humidity_rolling_3', 'humidity_rolling_6',
            
            # Volatility features
            'moisture_rolling_std', 'temp_rolling_std', 'humidity_rolling_std',
            
            # Rate of change features
            'moisture_change', 'moisture_change_2', 'temp_change', 'temp_change_2',
            'humidity_change', 'humidity_change_2',
            
            # Interaction and ratio features
            'temp_humidity_interaction', 'moisture_temp_ratio', 'moisture_humidity_ratio', 'temp_humidity_ratio',
            
            # Polynomial features
            'moisture_squared', 'temp_squared', 'humidity_squared',
            
            # Composite indices
            'dryness_index', 'moisture_deficit', 'heat_stress_index', 'humidity_stress_index',
            
            # Stability features
            'moisture_stability', 'temp_stability', 'humidity_stability',
            
            # Risk indicators
            'critical_low_risk', 'critical_high_risk'
        ]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['status'].copy()
        
        # Remove rows with NaN values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 30:  # Reduced minimum since we have more features
            logger.error(f"After feature engineering, only {len(X)} valid samples remain. At least 30 are needed.")
            raise ValueError(f"After feature engineering, only {len(X)} valid samples remain. At least 30 are needed.")
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Feature selection using mutual information
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        # Encode target for feature selection
        temp_label_encoder = LabelEncoder()
        y_temp_encoded = temp_label_encoder.fit_transform(y)
        
        # Select top features
        k_best = min(25, len(feature_columns))  # Select top 25 features or all if less
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        X_selected = selector.fit_transform(X, y_temp_encoded)
        
        # Get selected feature names
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Ensemble approach with multiple models
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
        from sklearn.svm import SVC
        
        # Create multiple classifiers
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Create voting classifier
        model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get class names for reporting
        class_names = label_encoder.classes_
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Save model, scaler, label encoder, and feature selector
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        
        # Save feature selector and selected features
        SELECTOR_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_selector.pkl')
        SELECTED_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'selected_features.json')
        
        joblib.dump(selector, SELECTOR_PATH)
        with open(SELECTED_FEATURES_PATH, 'w') as f:
            json.dump(selected_features, f)
        
        logger.info(f"Model, scaler, label encoder, and feature selector saved successfully")
        
        # Save metrics with additional information
        metrics = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(class_report['weighted avg']['precision'] * 100, 2),
            'recall': round(class_report['weighted avg']['recall'] * 100, 2),
            'f1_score': round(class_report['weighted avg']['f1-score'] * 100, 2),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'selected_features_count': len(selected_features),
            'class_distribution': {cls: int(np.sum(y == cls)) for cls in class_names},
            'confusion_matrix': conf_matrix.tolist(),
            'selected_features': selected_features,
            'model_type': 'VotingClassifier (RF + GB + ET)',
            'class_wise_performance': {
                cls: {
                    'precision': round(class_report[cls]['precision'] * 100, 2),
                    'recall': round(class_report[cls]['recall'] * 100, 2),
                    'f1_score': round(class_report[cls]['f1-score'] * 100, 2)
                } for cls in class_names
            }
        }
        
        try:
            with open(METRICS_PATH, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved successfully: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        raise

# Load or train the model
try:
    SELECTOR_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_selector.pkl')
    SELECTED_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'selected_features.json')
    
    if (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and 
        os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(SELECTOR_PATH)):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        feature_selector = joblib.load(SELECTOR_PATH)
        
        # Load selected features
        if os.path.exists(SELECTED_FEATURES_PATH):
            with open(SELECTED_FEATURES_PATH, 'r') as f:
                selected_features = json.load(f)
        
        logger.info(f"Model, scaler, label encoder, and feature selector loaded successfully")
    else:
        logger.warning(f"Model files not found. Training a new model...")
        model, _ = train_model()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    model = None
    scaler = None
    label_encoder = None
    feature_selector = None
    selected_features = None

def predict_soil_status(soil_moisture, temperature, humidity, location=None):
    """
    Predict soil status and irrigation recommendation
    
    Args:
        soil_moisture: Current soil moisture percentage
        temperature: Current temperature in Celsius
        humidity: Current humidity percentage
        location: Optional location parameter
    
    Returns:
        dict: Contains soil_status, irrigation_recommendation, and confidence
    """
    try:
        # Rule-based prediction (primary method)
        rule_based_status = determine_soil_status(soil_moisture, humidity, temperature)
        rule_based_irrigation = get_irrigation_recommendation(rule_based_status)
        
        # If model is available, use it for enhanced prediction
        if (model is not None and scaler is not None and label_encoder is not None and 
            feature_selector is not None and selected_features is not None):
            now = datetime.now()
            
            # Create a comprehensive feature set
            input_data = pd.DataFrame({
                'soil_moisture_percent': [soil_moisture],
                'temperature_celsius': [temperature],
                'humidity_percent': [humidity],
                'hour_sin': [np.sin(2 * np.pi * now.hour / 24)],
                'hour_cos': [np.cos(2 * np.pi * now.hour / 24)],
                'day_of_year_sin': [np.sin(2 * np.pi * now.timetuple().tm_yday / 365)],
                'day_of_year_cos': [np.cos(2 * np.pi * now.timetuple().tm_yday / 365)],
                'month_sin': [np.sin(2 * np.pi * now.month / 12)],
                'month_cos': [np.cos(2 * np.pi * now.month / 12)],
                'moisture_lag_1': [soil_moisture],
                'moisture_lag_2': [soil_moisture],
                'moisture_lag_3': [soil_moisture],
                'temp_lag_1': [temperature],
                'temp_lag_2': [temperature],
                'humidity_lag_1': [humidity],
                'humidity_lag_2': [humidity],
                'moisture_rolling_3': [soil_moisture],
                'moisture_rolling_6': [soil_moisture],
                'moisture_rolling_12': [soil_moisture],
                'temp_rolling_3': [temperature],
                'temp_rolling_6': [temperature],
                'humidity_rolling_3': [humidity],
                'humidity_rolling_6': [humidity],
                'moisture_rolling_std': [0.1],  # Assume stable conditions
                'temp_rolling_std': [0.1],
                'humidity_rolling_std': [0.1],
                'moisture_change': [0.0],
                'moisture_change_2': [0.0],
                'temp_change': [0.0],
                'temp_change_2': [0.0],
                'humidity_change': [0.0],
                'humidity_change_2': [0.0],
                'temp_humidity_interaction': [temperature * humidity],
                'moisture_temp_ratio': [soil_moisture / (temperature + 1)],
                'moisture_humidity_ratio': [soil_moisture / (humidity + 1)],
                'temp_humidity_ratio': [temperature / (humidity + 1)],
                'moisture_squared': [soil_moisture ** 2],
                'temp_squared': [temperature ** 2],
                'humidity_squared': [humidity ** 2],
                'dryness_index': [(temperature / (humidity + 1)) / (soil_moisture + 1)],
                'moisture_deficit': [max(0, 60 - soil_moisture)],
                'heat_stress_index': [max(0, temperature - 30)],
                'humidity_stress_index': [abs(humidity - 60)],
                'moisture_stability': [10.0],  # Assume stable
                'temp_stability': [10.0],
                'humidity_stability': [10.0],
                'critical_low_risk': [float(soil_moisture < 30 and temperature > 30 and humidity < 50)],
                'critical_high_risk': [float(soil_moisture > 70 and temperature < 25 and humidity > 75)]
            })
            
            # Select only the features that were selected during training
            input_data_selected = input_data[selected_features]
            
            # Handle any missing columns
            for col in selected_features:
                if col not in input_data_selected.columns:
                    input_data_selected[col] = 0.0
            
            # Scale features
            input_scaled = scaler.transform(input_data_selected)
            
            # Make prediction
            prediction_encoded = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            ml_status = label_encoder.inverse_transform([prediction_encoded])[0]
            ml_irrigation = get_irrigation_recommendation(ml_status)
            
            # Get confidence (max probability)
            confidence = float(max(prediction_proba))
            
            # Use ML prediction if confidence is high, otherwise use rule-based
            if confidence > 0.7:  # Increased threshold for higher confidence
                final_status = ml_status
                final_irrigation = ml_irrigation
                method = "Enhanced ML Model"
            else:
                final_status = rule_based_status
                final_irrigation = rule_based_irrigation
                method = "Rule-based (Low ML Confidence)"
                confidence = 0.95  # High confidence for rule-based
            
        else:
            # Use rule-based prediction only
            final_status = rule_based_status
            final_irrigation = rule_based_irrigation
            method = "Rule-based"
            confidence = 0.95
        
        result = {
            'status': final_status,
            'irrigation_recommendation': final_irrigation,
            'confidence': round(confidence * 100, 1),
            'method': method,
            'input_values': {
                'soil_moisture': soil_moisture,
                'temperature': temperature,
                'humidity': humidity
            }
        }
        
        logger.info(f"Prediction made for {location or 'unknown location'}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise

def get_model_metrics():
    """Get model performance metrics"""
    try:
        logger.info(f"Loading metrics from {METRICS_PATH}")
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
                logger.info(f"Metrics loaded: {metrics}")
                return metrics
        logger.warning(f"Metrics file not found at {METRICS_PATH}")
        return {
            'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
            'training_samples': None, 'class_distribution': None
        }
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}", exc_info=True)
        return {
            'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
            'training_samples': None, 'class_distribution': None
        }

def get_irrigation_schedule_recommendation(soil_status, current_weather=None):
    """
    Get detailed irrigation schedule recommendation
    
    Args:
        soil_status: Current soil status
        current_weather: Optional weather information
    
    Returns:
        dict: Detailed irrigation schedule
    """
    schedules = {
        "Critical Low": {
            "urgency": "Immediate",
            "frequency": "Every 2-3 hours",
            "duration": "15-20 minutes per session",
            "water_amount": "Heavy watering",
            "monitoring": "Check every hour"
        },
        "Dry": {
            "urgency": "Within 2-4 hours",
            "frequency": "Every 4-6 hours",
            "duration": "10-15 minutes per session",
            "water_amount": "Moderate to heavy watering",
            "monitoring": "Check every 2-3 hours"
        },
        "Normal": {
            "urgency": "Within 6-12 hours",
            "frequency": "Every 12-24 hours",
            "duration": "5-10 minutes per session",
            "water_amount": "Light to moderate watering",
            "monitoring": "Check twice daily"
        },
        "Wet": {
            "urgency": "Not required",
            "frequency": "Monitor only",
            "duration": "No watering",
            "water_amount": "No watering",
            "monitoring": "Check daily"
        },
        "Critical High": {
            "urgency": "Stop all irrigation",
            "frequency": "No watering",
            "duration": "No watering",
            "water_amount": "No watering - risk of root rot",
            "monitoring": "Check for drainage issues"
        }
    }
    
    return schedules.get(soil_status, schedules["Normal"])