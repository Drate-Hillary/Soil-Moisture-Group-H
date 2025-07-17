import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import requests
from datetime import datetime, timedelta
import os
from django.conf import settings
from django.db import connection
import logging
import pickle

logger = logging.getLogger(__name__)

class SoilMoisturePredictor:
    def __init__(self, model_type='classifier'):
        """
        Initialize the SoilMoistureClassifier.
        
        Args:
            model_type (str): 'classifier' for category prediction or 'regressor' for continuous values
        """
        self.model_type = model_type
        self.model_path = os.path.join(settings.BASE_DIR, 'models', f'soil_moisture_{model_type}.pkl')
        self.scaler_path = os.path.join(settings.BASE_DIR, 'models', f'scaler_{model_type}.pkl')
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'temperature_celsius', 'humidity_percent', 'precipitation', 
            'month', 'day', 'hour', 'previous_moisture'
        ]
        
        # Define moisture categories
        self.moisture_categories = {
            'Very Low': (0, 20),
            'Low': (20, 40),
            'Moderate': (40, 60),
            'High': (60, 80),
            'Very High': (80, 100)
        }
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")
            self.model = None
            self.scaler = None

    def categorize_moisture(self, moisture_value):
        """Convert continuous moisture value to category"""
        for category, (min_val, max_val) in self.moisture_categories.items():
            if min_val <= moisture_value < max_val:
                return category
        return 'Very High'  # For values >= 80

    def get_category_midpoint(self, category):
        """Get the midpoint value for a moisture category"""
        if category in self.moisture_categories:
            min_val, max_val = self.moisture_categories[category]
            return (min_val + max_val) / 2
        return 50  # Default fallback

    def load_model(self):
        """Load trained model and scaler from disk"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Model and scaler loaded successfully")
        else:
            logger.warning("Model or scaler files not found")
    
    def save_model(self):
        """Save model and scaler to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Model and scaler saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_uploaded_model(self, uploaded_file):
        """Load model from uploaded .pkl file"""
        try:
            # Save uploaded file temporarily
            temp_path = os.path.join(settings.BASE_DIR, 'models', 'temp_model.pkl')
            with open(temp_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Load and validate the model
            with open(temp_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's a complete model package or just the model
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_columns = model_data.get('feature_columns', self.feature_columns)
                self.model_type = model_data.get('model_type', self.model_type)
            else:
                # Assume it's just the model
                self.model = model_data
                # Determine model type based on the loaded model
                if hasattr(self.model, 'predict_proba'):
                    self.model_type = 'classifier'
                else:
                    self.model_type = 'regressor'
                # We'll need to retrain the scaler with existing data
                self.scaler = None
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Save to permanent location
            self.save_model()
            
            logger.info(f"Uploaded {self.model_type} model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading uploaded model: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def get_historical_data_from_db(self, location=None, days_back=365):
        """Fetch historical data from soil_moisture_records table"""
        try:
            from admindashboard.models import SoilMoistureRecord
            
            # Get data from the last year by default
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = SoilMoistureRecord.objects.filter(
                timestamp__gte=cutoff_date
            ).order_by('timestamp')
            
            if location:
                query = query.filter(location=location)
            
            # Convert to DataFrame
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

    def preprocess_data(self, df):
        """Preprocess data for training/prediction"""
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['location', 'timestamp'])
        
        # Feature engineering
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        
        # Create previous moisture feature (lag feature)
        df['previous_moisture'] = df.groupby('location')['soil_moisture_percent'].shift(1)
        
        # Add precipitation feature (you might want to fetch this from weather API)
        df['precipitation'] = 0  # Default to 0, can be enhanced with real data
        
        # Handle missing values
        df['previous_moisture'] = df['previous_moisture'].fillna(df['soil_moisture_percent'])
        
        # Remove rows with missing target values
        df = df.dropna(subset=['soil_moisture_percent'])
        
        # Convert continuous moisture to categories for classification
        if self.model_type == 'classifier':
            df['moisture_category'] = df['soil_moisture_percent'].apply(self.categorize_moisture)
        
        return df

    def train_model_with_db_data(self, location=None, retrain=False):
        """Train model using historical data from database"""
        try:
            # Get historical data
            df = self.get_historical_data_from_db(location=location)
            
            if df.empty:
                raise ValueError("No historical data available for training")
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            if len(df) < 10:
                raise ValueError("Insufficient data for training (minimum 10 records required)")
            
            # Prepare features and target
            X = df[self.feature_columns]
            
            if self.model_type == 'classifier':
                y = df['moisture_category']  # Use categorical target
            else:
                y = df['soil_moisture_percent']  # Use continuous target
            
            # Split data
            if self.model_type == 'classifier':
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Scale features
            if self.scaler is None or retrain:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
            
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if self.model is None or retrain:
                if self.model_type == 'classifier':
                    self.model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        class_weight='balanced'  # Handle class imbalance
                    )
                else:
                    self.model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            if self.model_type == 'classifier':
                metrics = {
                    'train_accuracy': accuracy_score(y_train, y_pred_train),
                    'test_accuracy': accuracy_score(y_test, y_pred_test),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'classes': list(self.model.classes_),
                    'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
                }
            else:
                metrics = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            
            # Save model
            self.save_model()
            
            logger.info(f"Model trained successfully: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict_single(self, location, temperature, humidity, current_moisture, 
                      timestamp=None, precipitation=0, return_probabilities=False):
        """Make single prediction"""
        if not self.model or not self.scaler:
            raise ValueError("Model not trained or loaded")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Prepare input data
        input_data = {
            'temperature_celsius': temperature,
            'humidity_percent': humidity,
            'precipitation': precipitation,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'previous_moisture': current_moisture
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select and order features
        X = input_df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        if self.model_type == 'classifier':
            # For classification models
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                prob_dict = dict(zip(self.model.classes_, probabilities))
                return {
                    'predicted_category': prediction,
                    'predicted_moisture_value': self.get_category_midpoint(prediction),
                    'probabilities': prob_dict,
                    'confidence': max(probabilities)
                }
            else:
                return {
                    'predicted_category': prediction,
                    'predicted_moisture_value': self.get_category_midpoint(prediction),
                    'confidence': 0.8  # Default confidence for classification without probabilities
                }
        else:
            # For regression models
            predicted_category = self.categorize_moisture(prediction)
            return {
                'predicted_category': predicted_category,
                'predicted_moisture_value': prediction,
                'confidence': 0.8  # Default confidence for regression
            }

    def get_weather_forecast(self, location="Kampala", days=7):
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
            
            for item in data['list'][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                forecasts.append({
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'precipitation': item.get('rain', {}).get('3h', 0)
                })
            
            return pd.DataFrame(forecasts)
            
        except Exception as e:
            logger.error(f"Weather API error: {str(e)}")
            # Return default forecast if API fails
            base_time = datetime.now()
            default_forecasts = []
            for i in range(days):
                default_forecasts.append({
                    'datetime': base_time + timedelta(days=i),
                    'temperature': 25.0,  # Default temperature
                    'humidity': 70.0,     # Default humidity
                    'precipitation': 0.0   # Default precipitation
                })
            return pd.DataFrame(default_forecasts)

    def predict_future_moisture(self, location, current_moisture, temperature, humidity, days=7):
        """Predict soil moisture categories for the next several days"""
        try:
            if not self.model or not self.scaler:
                raise ValueError("Model not trained or loaded")
            
            # Get weather forecast
            weather_df = self.get_weather_forecast("Kampala", days)
            
            predictions = []
            moisture_value = current_moisture
            
            for _, row in weather_df.iterrows():
                # Determine if we should return probabilities based on model type
                return_probs = (self.model_type == 'classifier' and hasattr(self.model, 'predict_proba'))
                
                prediction_result = self.predict_single(
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
                
                # Add probabilities if available
                if 'probabilities' in prediction_result:
                    prediction_dict['probabilities'] = {k: round(v, 3) for k, v in prediction_result['probabilities'].items()}
                
                predictions.append(prediction_dict)
                
                # Update moisture for next prediction using the predicted value
                moisture_value = prediction_result['predicted_moisture_value']
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future moisture: {str(e)}")
            raise

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.model:
            return None
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, importances))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance

    def get_model_info(self):
        """Get information about the current model"""
        if not self.model:
            return None
        
        info = {
            'model_type': self.model_type,
            'sklearn_model_type': type(self.model).__name__,
            'feature_columns': self.feature_columns,
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'moisture_categories': self.moisture_categories,
            'has_predict_proba': hasattr(self.model, 'predict_proba')
        }
        
        if self.model:
            if hasattr(self.model, 'classes_'):
                info['classes'] = list(self.model.classes_)
            info['feature_importance'] = self.get_feature_importance()
        
        return info