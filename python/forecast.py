import sys
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    # Handle TensorFlow version differences for optimizers
    try:
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        from keras.optimizers import Adam
    # Handle TensorFlow version differences for callbacks
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        from keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    # Suppress TensorFlow info messages and optimize for speed
    tf.get_logger().setLevel('ERROR')
    # Optimize TensorFlow for faster execution
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # Disable GPU if available to avoid initialization overhead
    tf.config.set_visible_devices([], 'GPU')
    # Pre-initialize TensorFlow to avoid timeout issues
    print("TensorFlow initializing...", file=sys.stderr)
    # Test TensorFlow with a simple operation
    tf.constant(1.0)
    print("TensorFlow ready", file=sys.stderr)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using scikit-learn models only", file=sys.stderr)
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow initialization failed: {e}, using scikit-learn models only", file=sys.stderr)

"""
Enhanced AI-powered forecasting script
Reads stdin lines in format: YYYY-MM-DD,revenue
Outputs JSON with advanced forecast and chart-friendly data
"""

def create_lstm_model(input_shape, sequence_length=7):
    """Create LSTM neural network for time series forecasting"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, input_shape)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def create_dense_model(input_shape):
    """Create dense neural network for forecasting"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def prepare_sequence_data(X, y, sequence_length=7):
    """Prepare data for LSTM sequence modeling"""
    if len(X) < sequence_length:
        return None, None
    
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def train_tensorflow_model(model, X_train, y_train, X_test=None, y_test=None, epochs=20, batch_size=16):
    """Train TensorFlow model with early stopping"""
    if not TENSORFLOW_AVAILABLE or model is None:
        return None, 0
    
    try:
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        callbacks = [early_stopping] if X_test is not None else []
        
        # Train the model
        if X_test is not None and y_test is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            # Get validation loss for scoring
            val_loss = min(history.history['val_loss'])
            score = max(0, 1 - val_loss / (np.mean(y_train) + 1))  # Convert to R²-like score
        else:
            history = model.fit(
                X_train, y_train,
                epochs=min(epochs, 20),  # Fewer epochs for small datasets
                batch_size=batch_size,
                verbose=0
            )
            # Use training loss for scoring
            train_loss = min(history.history['loss'])
            score = max(0, 1 - train_loss / (np.mean(y_train) + 1))
        
        return model, score
        
    except Exception as e:
        print(f"TensorFlow training error: {e}", file=sys.stderr)
        return None, 0

def main():
    try:
        data = sys.stdin.read().strip().splitlines()
        if not data:
            print(json.dumps({
                'forecast': None,
                'confidence': 0,
                'trend': 'stable',
                'chart': { 'labels': [], 'datasets': [] },
                'insights': []
            }))
            return

        # Parse input data - only include rows with actual sales (revenue > 0)
        rows = [line.split(',') for line in data if ',' in line]
        df = pd.DataFrame(rows, columns=['date','revenue'])
        df['date'] = pd.to_datetime(df['date'])
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        
        # Filter out zero-revenue entries BEFORE processing (only keep days with actual sales)
        df = df[df['revenue'] > 0].copy()
        df = df.sort_values('date')
        df['t'] = np.arange(len(df))
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        # Advanced feature engineering
        df['revenue_ma7'] = df['revenue'].rolling(window=7, min_periods=1).mean()
        df['revenue_ma30'] = df['revenue'].rolling(window=30, min_periods=1).mean()
        df['revenue_diff'] = df['revenue'].diff().fillna(0)
        df['revenue_pct_change'] = df['revenue'].pct_change().fillna(0)
        
        # Additional features for better accuracy
        df['revenue_std'] = df['revenue'].rolling(window=7, min_periods=1).std().fillna(0)
        df['revenue_trend'] = df['revenue'].rolling(window=3, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['date'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 25).astype(int)

        # Prepare features for machine learning
        feature_cols = ['t', 'day_of_week', 'month', 'revenue_ma7', 'revenue_ma30', 'revenue_diff', 
                        'revenue_std', 'revenue_trend', 'is_weekend', 'is_month_start', 'is_month_end']
        X = df[feature_cols].values
        y = df['revenue'].values

        # Train multiple models
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Add TensorFlow models if available and data is sufficient
        if TENSORFLOW_AVAILABLE and len(X) >= 3:  # Reduced minimum data requirement
            # Only add TensorFlow models for larger datasets to avoid overhead
            if len(X) >= 5:  # Reduced from 10 to 5
                models['dense_nn'] = create_dense_model(X.shape[1])
            # Only add LSTM if we have enough data for sequences
            if len(X) >= 7:  # Reduced from 14 to 7
                models['lstm'] = create_lstm_model(X.shape[1])

        best_model = None
        best_score = -float('inf')
        best_model_name = None

        # Model training and selection
        for name, model in models.items():
            try:
                # Handle TensorFlow models differently
                if name in ['dense_nn', 'lstm'] and TENSORFLOW_AVAILABLE:
                    if len(X) < 5:
                        # For very small datasets, skip TensorFlow models
                        continue
                    
                    # Prepare data for TensorFlow
                    if name == 'lstm':
                        # Prepare sequence data for LSTM
                        X_seq, y_seq = prepare_sequence_data(X, y)
                        if X_seq is None:
                            continue
                        
                        # Use last 20% for validation
                        split_idx = int(len(X_seq) * 0.8)
                        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                    else:
                        # Dense neural network
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train TensorFlow model
                    trained_model, score = train_tensorflow_model(model, X_train, y_train, X_test, y_test)
                    if trained_model is not None:
                        model = trained_model  # Update model reference
                else:
                    # Handle scikit-learn models
                    if len(X) < 3:  # Reduced minimum data requirement
                        model.fit(X, y)
                        # Use a simple score based on data quality
                        avg_revenue = np.mean(y) if len(y) > 0 else 0
                        score = min(0.8, avg_revenue / 1000)  # Simple scoring based on revenue
                    else:
                        # Use last 20% for validation
                        split_idx = int(len(X) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate R² score
                        score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error training {name}: {e}", file=sys.stderr)
                continue

        # Generate forecast
        if best_model is not None:
            # Predict next 7 days
            predictions = []
            confidence_scores = []
            
            # Check if we're using a TensorFlow model
            is_tensorflow_model = hasattr(best_model, 'predict') and hasattr(best_model, 'layers')
            
            if is_tensorflow_model and best_model_name == 'lstm':
                # LSTM prediction - use sequence data
                last_sequence = X[-7:]  # Use last 7 days as sequence
                if len(last_sequence) < 7:
                    # Pad with zeros if not enough data
                    padding = np.zeros((7 - len(last_sequence), X.shape[1]))
                    last_sequence = np.vstack([padding, last_sequence])
                
                for i in range(7):
                    # Reshape for LSTM input
                    sequence_input = last_sequence.reshape(1, 7, X.shape[1])
                    pred = float(best_model.predict(sequence_input, verbose=0)[0][0])
                    predictions.append(max(0.0, round(pred, 2)))
                    
                    # Update sequence for next prediction
                    new_features = last_sequence[-1].copy()
                    new_features[0] += 1  # increment time
                    new_features[1] = (new_features[1] + 1) % 7  # next day of week
                    if new_features[1] == 0:  # if it's Monday, increment month
                        new_features[2] = (new_features[2] % 12) + 1
                    
                    # Shift sequence and add new prediction
                    last_sequence = np.vstack([last_sequence[1:], new_features])
                    
            else:
                # Standard prediction for scikit-learn and dense neural networks
                last_features = X[-1:].copy()
                
                for i in range(7):
                    if is_tensorflow_model:
                        pred = float(best_model.predict(last_features, verbose=0)[0][0])
                    else:
                        pred = float(best_model.predict(last_features)[0])
                    predictions.append(max(0.0, round(pred, 2)))
                    
                    # Update features for next prediction
                    last_features[0][0] += 1  # increment time
                    last_features[0][1] = (last_features[0][1] + 1) % 7  # next day of week
                    if last_features[0][1] == 0:  # if it's Monday, increment month
                        last_features[0][2] = (last_features[0][2] % 12) + 1
            
            # Calculate confidence scores for all predictions
            for i in range(7):
                # Calculate confidence based on model performance and data quality
                base_confidence = max(0, min(100, round((best_score + 1) * 50, 1)))
                
                # Adjust confidence based on data variance and amount
                data_variance = np.var(y) if len(y) > 1 else 0
                avg_revenue = np.mean(y) if len(y) > 0 else 0
                
                # Lower confidence for low revenue or high variance
                variance_factor = max(0.5, 1 - (data_variance / (avg_revenue + 1)))
                revenue_factor = max(0.3, min(1.0, avg_revenue / 1000))  # Scale based on revenue level
                
                # TensorFlow models get slight confidence boost
                if is_tensorflow_model:
                    base_confidence *= 1.1
                
                confidence = max(10, min(100, round(base_confidence * variance_factor * revenue_factor, 1)))
                confidence_scores.append(confidence)

            # Calculate overall confidence
            avg_confidence = np.mean(confidence_scores)
            
            # Determine trend
            current_avg = np.mean(y[-7:]) if len(y) >= 7 else np.mean(y)
            future_avg = np.mean(predictions)
            trend = 'increasing' if future_avg > current_avg * 1.05 else \
                   'decreasing' if future_avg < current_avg * 0.95 else 'stable'

            # Generate AI insights
            insights = []
            
            # Add model type information
            if is_tensorflow_model:
                if best_model_name == 'lstm':
                    insights.append("Using advanced LSTM neural network for time series forecasting.")
                else:
                    insights.append("Using deep neural network for enhanced pattern recognition.")
            else:
                insights.append(f"Using {best_model_name} model for revenue forecasting.")
            
            if trend == 'increasing':
                insights.append("Positive growth trend detected. Consider increasing staffing for peak periods.")
            elif trend == 'decreasing':
                insights.append("Declining trend observed. Review pricing strategy and service offerings.")
            else:
                insights.append("Stable business performance. Maintain current operations.")

            # Add seasonal insights
            if df['month'].iloc[-1] in [11, 12, 1, 2]:  # Holiday season
                insights.append("Holiday season approaching. Consider promotional campaigns.")
            elif df['month'].iloc[-1] in [6, 7, 8]:  # Summer season
                insights.append("Summer season peak. Focus on hair and nail services.")

            # Add inventory insights
            if len(y) > 0:
                recent_growth = ((y[-1] - y[0]) / y[0] * 100) if y[0] > 0 else 0
                if recent_growth > 10:
                    insights.append("Strong growth detected. Review inventory levels to meet demand.")

        else:
            # Fallback to simple linear regression
            X_simple = df[['t']].values
            y_simple = df['revenue'].values
            model = LinearRegression()
            model.fit(X_simple, y_simple)
            
            next_t = np.array([[df['t'].max() + 1]])
            pred = float(model.predict(next_t)[0])
            predictions = [max(0.0, round(pred, 2))] * 7
            # Calculate confidence based on data quality for fallback
            avg_revenue = np.mean(y) if len(y) > 0 else 0
            data_variance = np.var(y) if len(y) > 1 else 0
            
            # Lower confidence for fallback model
            base_confidence = 50.0
            variance_factor = max(0.3, 1 - (data_variance / (avg_revenue + 1)))
            revenue_factor = max(0.2, min(0.8, avg_revenue / 1000))
            
            avg_confidence = max(15, min(75, round(base_confidence * variance_factor * revenue_factor, 1)))
            trend = 'stable'
            insights = ["Using basic forecasting model. More data needed for advanced AI predictions."]

        # Prepare chart data - only show actual for days with REAL sales data (> 0)
        # Filter to only include dates with actual sales (revenue > 0)
        actual_sales_mask = df['revenue'] > 0
        actual_dates = df[actual_sales_mask]['date']
        actual_revenue = df[actual_sales_mask]['revenue'].values
        
        # Get the last actual sales date (not just last date in dataframe)
        last_actual_date = actual_dates.iloc[-1] if len(actual_dates) > 0 else df['date'].iloc[-1]
        
        # Check if today has sales data - if so, include it in actual, forecasts start tomorrow
        today = pd.Timestamp.now().normalize()  # Get today's date at midnight
        if last_actual_date.date() >= today.date():
            # Last actual date is today or future, forecasts start tomorrow
            forecast_start = today + pd.Timedelta(days=1)
        else:
            # Last actual date is in the past, forecasts start the day after
            forecast_start = last_actual_date + pd.Timedelta(days=1)
        
        # Create labels and data arrays - ONLY include dates with actual sales, then forecast dates
        # This ensures no gaps are shown where there's no actual data
        # Format labels as YYYY-MM-DD for proper parsing in JavaScript
        actual_labels = [d.strftime('%Y-%m-%d') for d in actual_dates]
        
        # Forecast dates start from forecast_start
        forecast_dates = [(forecast_start + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        
        # Combine labels: actual dates with sales + forecast dates
        all_labels = actual_labels + forecast_dates
        
        # Prepare actual data: real values for actual sales dates, then None for all forecast dates
        # None will be converted to null in JSON, which Chart.js treats as missing data (breaks line)
        actual_data = [float(v) for v in actual_revenue] + [None] * 7
        
        # Prepare predicted data: None for all actual sales dates, then predictions for forecast dates
        # This ensures predictions only appear AFTER the last actual sales date
        predicted_data = [None] * len(actual_revenue) + [float(x) if x > 0 else None for x in predictions]
        
        # IMPORTANT: Ensure None becomes null in JSON (Chart.js will treat null as gap and break line)

        result = {
            'forecast': {
                'next_7_days': [float(x) for x in predictions],
                'total_predicted': float(round(sum(predictions), 2)),
                'average_daily': float(round(np.mean(predictions), 2))
            },
            'confidence': float(round(avg_confidence, 1)),
            'trend': trend,
            'chart': {
                'labels': all_labels,
                'datasets': [
                    {
                        'label': 'Actual',
                        'data': actual_data,
                        'borderColor': '#0d6efd',
                        'backgroundColor': 'rgba(13, 110, 253, 0.1)',
                        'fill': False
                    },
                    {
                        'label': 'Predicted',
                        'data': predicted_data,
                        'borderColor': '#6610f2',
                        'backgroundColor': 'rgba(102, 16, 242, 0.1)',
                        'borderDash': [5, 5],
                        'fill': False
                    }
                ]
            },
            'insights': insights,
            'statistics': {
                'historical_average': float(round(np.mean(y), 2)) if len(y) > 0 else 0.0,
                'historical_max': float(round(np.max(y), 2)) if len(y) > 0 else 0.0,
                'historical_min': float(round(np.min(y), 2)) if len(y) > 0 else 0.0,
                'data_points': int(len(y))
            }
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            'error': 'Forecast generation failed',
            'detail': str(e),
            'forecast': None,
            'confidence': 0,
            'trend': 'unknown',
            'chart': { 'labels': [], 'datasets': [] },
            'insights': ['Error in AI analysis. Using basic forecasting.']
        }))

if __name__ == '__main__':
    main()
