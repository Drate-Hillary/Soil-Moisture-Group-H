import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
import joblib
import requests
from datetime import datetime, timedelta, timezone
import os
import logging
import pickle

logger = logging.getLogger(__name__)

# Global variables
model_type = 'regressor'
model_path = os.path.join('models', f'soil_moisture_{model_type}.pkl')
scaler_path = os.path.join('models', f'scaler_{model_type}.pkl')
model = None
scaler = None

# Updated feature columns to match what's actually available after preprocessing
feature_columns = [
    'temperature_celsius', 'humidity_percent', 'precipitation',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    'day_of_week', 'day_of_year', 'previous_moisture',
    'moisture_change', 'rolling_7d_avg', 'temp_humidity_interaction'
]

selected_features = feature_columns.copy()  # Will be updated after feature selection
moisture_categories = {
    'Very Low': (0, 20),
    'Low': (20, 40),
    'Moderate': (40, 60),
    'High': (60, 80),
    'Very High': (80, 100)
}
k_features = min(8, len(feature_columns))  # Increased to accommodate more features

# Create models directory
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Try to load model at startup
try:
    load_model()
except Exception as e:
    logger.warning(f"Could not load model: {str(e)}")
    model = None
    scaler = None

def categorize_moisture(moisture_value):
    """Convert continuous moisture value to category"""
    for category, (min_val, max_val) in moisture_categories.items():
        if min_val <= moisture_value < max_val:
            return category
    return 'Very High'  # For values >= 80

def get_category_midpoint(category):
    """Get the midpoint value for a moisture category"""
    if category in moisture_categories:
        min_val, max_val = moisture_categories[category]
        return (min_val + max_val) / 2
    return 50  # Default fallback

def load_model():
    """Load trained model pipeline and scaler from disk"""
    global model, scaler, selected_features
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            # Extract selected features from the pipeline
            if isinstance(model, Pipeline):
                select_k_best = model.named_steps.get('selectkbest')
                if select_k_best:
                    support = select_k_best.get_support()
                    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if support[i]]
                else:
                    selected_features = feature_columns.copy()
            # Validate model type
            if not isinstance(model, Pipeline):
                logger.error(f"Loaded model is of type {type(model).__name__}, expected Pipeline")
                model = None
                scaler = None
                return False
            logger.info("Model and scaler loaded successfully")
            return True
        else:
            logger.warning("Model or scaler files not found")
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import RobustScaler
            # Initialize new pipeline
            scaler = StandardScaler()
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', RobustScaler()),
                ('selectkbest', SelectKBest(score_func=f_regression, k=k_features)),
                ('regressor', VotingRegressor([
                    ('rf', RandomForestRegressor(
                        n_estimators=200, max_depth=10, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingRegressor(
                        n_estimators=200, max_depth=5, min_samples_split=5,
                        min_samples_leaf=2, random_state=42)),
                    ('et', ExtraTreesRegressor(
                        n_estimators=200, max_depth=10, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1))
                ]))
            ])
            selected_features = feature_columns.copy()  # Reset to all features until trained
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        scaler = None
        selected_features = feature_columns.copy()
        return False

def save_model():
    """Save model and scaler to disk"""
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info("Model and scaler saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_uploaded_model(uploaded_file):
    """Load model from uploaded .pkl file"""
    global model, scaler, feature_columns, model_type
    try:
        temp_path = os.path.join('models', 'temp_model.pkl')
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        with open(temp_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            feature_columns = model_data.get('feature_columns', feature_columns)
            model_type = model_data.get('model_type', model_type)
        else:
            model = model_data
            if hasattr(model, 'predict_proba'):
                model_type = 'classifier'
            else:
                model_type = 'regressor'
            scaler = None
        
        os.remove(temp_path)
        save_model()
        logger.info(f"Uploaded {model_type} model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading uploaded model: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def get_historical_data_from_db(location=None, days_back=365):
    """Fetch historical data from soil_moisture_records table"""
    try:
        from admindashboard.models import SoilMoistureRecord
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        query = SoilMoistureRecord.objects.filter(
            timestamp__gte=cutoff_date,
            soil_moisture_percent__isnull=False,
            temperature_celsius__isnull=False,
            humidity_percent__isnull=False
        ).order_by('timestamp')
        
        # Add outlier filtering
        query = query.filter(
            soil_moisture_percent__gte=0,
            soil_moisture_percent__lte=100,
            temperature_celsius__gte=-20,
            temperature_celsius__lte=60,
            humidity_percent__gte=0,
            humidity_percent__lte=100
        )
        
        if location:
            query = query.filter(location=location)
        
        data = []
        for record in query:
            data.append({
                'timestamp': record.timestamp,
                'location': record.location,
                'soil_moisture_percent': record.soil_moisture_percent,
                'temperature_celsius': record.temperature_celsius,
                'humidity_percent': record.humidity_percent,
                'sensor_id': record.sensor_id,
                'status': record.status
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No historical data found in database")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df, is_prediction=False):
    """Preprocess data for training or prediction, handling NaN values robustly"""
    if df.empty:
        return df

    # Ensure precipitation column exists
    if 'precipitation' not in df.columns:
        df['precipitation'] = 0.0

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['location', 'timestamp'])

    # Create temporal features
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear

    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Handle lag and rolling features
    if is_prediction and len(df) == 1:
        # For single-row prediction, use provided soil_moisture_percent as previous_moisture
        df['previous_moisture'] = df['soil_moisture_percent']
        df['moisture_change'] = 0.0
        df['rolling_7d_avg'] = df['soil_moisture_percent']
    else:
        # Lag features
        df['previous_moisture'] = df.groupby('location')['soil_moisture_percent'].shift(1)
        df['moisture_change'] = df.groupby('location')['soil_moisture_percent'].diff()
        # Rolling statistics
        df['rolling_7d_avg'] = df.groupby('location')['soil_moisture_percent'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

    # Weather interactions
    df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']

    # Robust NaN handling
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0.0)

    # Drop original temporal features
    columns_to_drop = ['month', 'day', 'hour']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(existing_columns_to_drop, axis=1)

    if model_type == 'classifier':
        df['moisture_category'] = df['soil_moisture_percent'].apply(categorize_moisture)

    return df

def train_model_with_db_data(location=None, retrain=False):
    """Train model using historical data from database with comprehensive metrics"""
    global model, scaler, selected_features
    try:
        df = get_historical_data_from_db(location=location)
        if df.empty:
            raise ValueError("No historical data available for training")
        
        df = preprocess_data(df)
        
        # Ensure we have the required features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                df[col] = 0.0
        
        # Prepare features and target
        X = df[feature_columns]
        if model_type == 'classifier':
            y = df['moisture_category']
        else:
            y = df['soil_moisture_percent']
        
        # Add time-based validation split
        if len(df) > 100:  # Only if sufficient data
            # Sort by timestamp first
            sorted_indices = df.sort_values('timestamp').index
            split_idx = int(0.8 * len(df))
            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]
            
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if model_type == 'classifier' else None
            )
        
        if model is None or retrain:
            if model_type == 'regressor':
                base_models = [
                    ('rf', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1
                    )),
                    ('gb', GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=4,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        subsample=0.8,
                        random_state=42
                    )),
                    ('et', ExtraTreesRegressor(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1
                    ))
                ]
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('selectkbest', SelectKBest(score_func=f_regression, k='all')),
                    ('regressor', VotingRegressor(
                        estimators=base_models,
                        weights=None,
                        n_jobs=-1
                    ))
                ])
                
                # Add cross-validation for hyperparameter tuning
                from sklearn.model_selection import GridSearchCV
                param_grid = {
                    'selectkbest__k': [min(5, len(feature_columns)), min(8, len(feature_columns)), 'all'],
                    'regressor__weights': [
                        None,
                        [1, 1, 1],
                        [2, 1, 1]
                    ]
                }
                
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best params: {grid_search.best_params_}")
            else:
                from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('selectkbest', SelectKBest(score_func=f_classif, k=k_features)),
                    ('classifier', VotingClassifier([
                        ('rf', RandomForestClassifier(
                            n_estimators=200, max_depth=10, min_samples_split=5,
                            min_samples_leaf=2, random_state=42, n_jobs=-1,
                            class_weight='balanced')),
                        ('gb', GradientBoostingClassifier(
                            n_estimators=200, max_depth=5, min_samples_split=5,
                            min_samples_leaf=2, random_state=42)),
                        ('et', ExtraTreesClassifier(
                            n_estimators=200, max_depth=10, min_samples_split=5,
                            min_samples_leaf=2, random_state=42, n_jobs=-1,
                            class_weight='balanced'))
                    ]))
                ])
            scaler = model.named_steps['scaler']
        
        # Fit the pipeline
        model.fit(X_train, y_train)
        
        # Update selected features
        select_k_best = model.named_steps['selectkbest']
        support = select_k_best.get_support()
        selected_features = [feature_columns[i] for i in range(len(feature_columns)) if support[i]]
        logger.info(f"Selected features: {selected_features}")
        
        # Generate predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics based on model type
        if model_type == 'regressor':
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_evs = explained_variance_score(y_train, y_pred_train)
            test_evs = explained_variance_score(y_test, y_pred_test)
            
            metrics = {
                'train_rmse': round(train_rmse, 4),
                'test_rmse': round(test_rmse, 4),
                'train_mae': round(train_mae, 4),
                'test_mae': round(test_mae, 4),
                'train_r2': round(train_r2, 4),
                'test_r2': round(test_r2, 4),
                'train_evs': round(train_evs, 4),
                'test_evs': round(test_evs, 4),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': 'regressor',
                'selected_features': selected_features,
                'feature_scores': dict(zip(feature_columns, select_k_best.scores_))
            }
            
            print("\nRegression Metrics:")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Training MAE: {train_mae:.4f}")
            print(f"Test MAE: {test_mae:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Training Explained Variance: {train_evs:.4f}")
            print(f"Test Explained Variance: {test_evs:.4f}")
        else:
            from sklearn.metrics import (accuracy_score, precision_score, 
                                       recall_score, f1_score, classification_report,
                                       confusion_matrix)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            metrics = {
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'train_precision': round(train_precision, 4),
                'test_precision': round(test_precision, 4),
                'train_recall': round(train_recall, 4),
                'test_recall': round(test_recall, 4),
                'train_f1_score': round(train_f1, 4),
                'test_f1_score': round(test_f1, 4),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': 'classifier',
                'selected_features': selected_features,
                'feature_scores': dict(zip(feature_columns, select_k_best.scores_)),
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
                'classes': list(model.named_steps['classifier'].classes_)
            }
            
            print("\nClassification Metrics:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Training Precision: {train_precision:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Training Recall: {train_recall:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Training F1 Score: {train_f1:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
        
        save_model()
        logger.info(f"Model trained successfully with metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def create_prediction_dataframe(location, temperature, humidity, current_moisture, timestamp=None, precipitation=0):
    """Create a properly formatted dataframe for prediction"""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create basic data
    data = {
        'timestamp': timestamp,
        'location': location,
        'temperature_celsius': temperature,
        'humidity_percent': humidity,
        'precipitation': precipitation,
        'soil_moisture_percent': current_moisture  # This will be used for lag features
    }
    
    df = pd.DataFrame([data])
    
    # Apply the same preprocessing as training data
    df = preprocess_data(df)
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    return df[feature_columns]

def predict_single(location, temperature, humidity, current_moisture, timestamp=None, precipitation=0, return_probabilities=False):
    """Make single prediction"""
    if not model:
        raise ValueError("Model not trained or loaded")
    
    # Create properly formatted input data
    input_df = create_prediction_dataframe(
        location, temperature, humidity, current_moisture, timestamp, precipitation
    )
    
    prediction = model.predict(input_df)[0]
    
    if model_type == 'regressor':
        predicted_category = categorize_moisture(prediction)
        return {
            'predicted_category': predicted_category,
            'predicted_moisture_value': round(prediction, 2),
            'confidence': 0.8  # VotingRegressor doesn't provide probabilities
        }
    else:
        # For classifier
        if return_probabilities and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            probabilities = dict(zip(classes, proba))
            return {
                'predicted_category': prediction,
                'predicted_moisture_value': round(get_category_midpoint(prediction), 2),
                'confidence': round(max(proba), 3),
                'probabilities': probabilities
            }
        else:
            return {
                'predicted_category': prediction,
                'predicted_moisture_value': round(get_category_midpoint(prediction), 2),
                'confidence': 0.8
            }

def get_weather_forecast(location="Kampala", days=7):
    """Fetch weather forecast from API"""
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            raise ValueError("OpenWeather API key not found")
        
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        forecasts = []
        for item in data['list'][:days * 8]:
            forecasts.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'precipitation': item.get('rain', {}).get('3h', 0)
            })
        
        return pd.DataFrame(forecasts)
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        base_time = datetime.now()
        default_forecasts = []
        for i in range(days):
            default_forecasts.append({
                'datetime': base_time + timedelta(days=i),
                'temperature': 25.0,
                'humidity': 70.0,
                'precipitation': 0.0
            })
        return pd.DataFrame(default_forecasts)

def predict_future_moisture(location, current_moisture, temperature, humidity, days=7):
    """Predict soil moisture categories for the next several days"""
    try:
        if not model or not scaler:
            raise ValueError("Model not trained or loaded")
        
        weather_df = get_weather_forecast("Kampala", days)
        predictions = []
        moisture_value = current_moisture
        
        for _, row in weather_df.iterrows():
            return_probs = (model_type == 'classifier' and hasattr(model, 'predict_proba'))
            prediction_result = predict_single(
                location=location,
                temperature=row['temperature'],
                humidity=row['humidity'],
                current_moisture=moisture_value,
                timestamp=row['datetime'],
                precipitation=row['precipitation'],
                return_probabilities=return_probs
            )
            
            prediction_dict = {
                'date': row['datetime'].strftime('%Y-%m-%d'),
                'datetime': row['datetime'],
                'predicted_category': prediction_result['predicted_category'],
                'predicted_moisture_value': round(prediction_result['predicted_moisture_value'], 2),
                'confidence': round(prediction_result['confidence'], 3),
                'temperature': round(row['temperature'], 2),
                'humidity': round(row['humidity'], 2),
                'precipitation': round(row['precipitation'], 2)
            }
            
            if 'probabilities' in prediction_result:
                prediction_dict['probabilities'] = {k: round(v, 3) for k, v in prediction_result['probabilities'].items()}
            
            predictions.append(prediction_dict)
            moisture_value = prediction_result['predicted_moisture_value']
        
        return predictions
    except Exception as e:
        logger.error(f"Error predicting future moisture: {str(e)}")
        raise

def get_feature_importance():
    """Get feature importance from trained model"""
    if not model:
        return None
    
    try:
        # Get the final estimator from the pipeline
        if hasattr(model, 'named_steps'):
            if 'regressor' in model.named_steps:
                estimator = model.named_steps['regressor']
            elif 'classifier' in model.named_steps:
                estimator = model.named_steps['classifier']
            else:
                return None
        else:
            estimator = model
        
        # Get feature importance
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        else:
            # For voting estimators, average the importance
            if hasattr(estimator, 'estimators_'):
                importances = []
                for est in estimator.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances.append(est.feature_importances_)
                if importances:
                    importances = np.mean(importances, axis=0)
                else:
                    return None
            else:
                return None
        
        # Map to selected features
        if len(importances) == len(selected_features):
            feature_importance = dict(zip(selected_features, importances))
        else:
            feature_importance = dict(zip(feature_columns[:len(importances)], importances))
        
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return sorted_importance
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return None
    
    
import matplotlib.pyplot as plt
def visualize_feature_importance():
    importance = get_feature_importance()
    if importance:
        plt.bar(importance.keys(), importance.values())
        plt.xticks(rotation=45)
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()