import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def create_synthetic_data(num_samples=2000):
    """
    Generates a synthetic dataset for hotel booking demand prediction.
    """
    np.random.seed(42)
    
    # --- Feature Generation ---
    
    # 1. Customer Type (Categorical)
    customer_types = ['Transient', 'Contract', 'Group', 'Transient-Party']
    data = {'customer_type': np.random.choice(customer_types, num_samples, p=[0.7, 0.1, 0.1, 0.1])}
    
    # 2. Location (Categorical)
    locations = ['City Center', 'Airport', 'Resort', 'Suburban']
    data['location'] = np.random.choice(locations, num_samples, p=[0.4, 0.2, 0.3, 0.1])
    
    # 3. Booking Behavior (Numeric)
    data['lead_time'] = np.random.randint(0, 400, num_samples)
    data['stays_in_weekend_nights'] = np.random.randint(0, 3, num_samples)
    data['stays_in_week_nights'] = np.random.randint(0, 6, num_samples)
    data['previous_cancellations'] = np.random.choice([0, 1, 2, 5], num_samples, p=[0.9, 0.05, 0.03, 0.02])
    data['adr'] = np.random.normal(150, 40, num_samples) # Average Daily Rate
    data['is_repeated_guest'] = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    
    df = pd.DataFrame(data)
    
    # --- Target Variable Creation ("is_high_demand") ---
    # This is the "secret" rule our AI will try to learn.
    # We define high demand based on a combination of factors.
    demand_score = 0
    
    # High ADR contributes
    demand_score += (df['adr'] - 150) / 40
    
    # Short lead time contributes
    demand_score += (50 - df['lead_time']) / 50 
    
    # Resort locations have higher demand
    demand_score += (df['location'] == 'Resort').astype(int) * 2.5
    
    # Groups have higher demand
    demand_score += (df['customer_type'] == 'Group').astype(int) * 1.5
    
    # Set the top 25% highest scores as "High Demand"
    demand_threshold = np.percentile(demand_score, 75)
    df['is_high_demand'] = (demand_score > demand_threshold).astype(int)
    
    # Clean up ADR to be realistic
    df['adr'] = df['adr'].clip(lower=40, upper=500).round(2)
    
    print(f"Generated {num_samples} synthetic bookings.")
    print(f"High Demand (1): {df['is_high_demand'].sum()} bookings")
    print(f"Low Demand (0):  {num_samples - df['is_high_demand'].sum()} bookings")
    print("-" * 30, "\n")
    
    return df

def train_demand_predictor(df):
    """
    Preprocesses data, trains a RandomForest model, and evaluates it.
    """
    
    # --- 1. Define Features (X) and Target (y) ---
    target_variable = 'is_high_demand'
    
    # Separate features from the target variable
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    
    # --- 2. Define Preprocessing Steps ---
    
    # Identify which columns are numeric and which are categorical
    numeric_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                        'previous_cancellations', 'adr', 'is_repeated_guest']
    
    categorical_features = ['customer_type', 'location']
    
    # Create a transformer for numeric features:
    # We will 'scale' them (StandardScaler) so that large numbers (like lead_time)
    # don't outweigh small numbers (like is_repeated_guest).
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create a transformer for categorical features:
    # We will 'one-hot encode' them, turning 'Location' into separate
    # columns like 'Location_Resort', 'Location_Airport', etc., with 0s or 1s.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine these transformers into a single 'preprocessor'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # --- 3. Create and Train the Model Pipeline ---
    
    # Now, create the full pipeline:
    # 1. It will preprocess the data (using our 'preprocessor')
    # 2. It will then feed the processed data into the classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training model on {len(X_train)} bookings...")
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    print("Model training complete.")
    print("-" * 30, "\n")
    
    # --- 4. Evaluate the Model ---
    print("--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Demand (0)', 'High Demand (1)']))
    
    return model_pipeline

def predict_new_bookings(model_pipeline):
    """
    Shows how to use the trained model to predict demand for new bookings.
    """
    print("\n" + "-" * 30)
    print("--- New Booking Predictions ---")
    
    # Create new sample data (as a DataFrame, matching the original columns)
    new_data = {
        'customer_type': ['Transient', 'Group', 'Contract'],
        'location': ['Resort', 'City Center', 'Airport'],
        'lead_time': [15, 90, 200],
        'stays_in_weekend_nights': [2, 2, 0],
        'stays_in_week_nights': [5, 3, 3],
        'previous_cancellations': [0, 1, 0],
        'adr': [250.0, 180.0, 95.0],
        'is_repeated_guest': [0, 0, 1]
    }
    new_bookings_df = pd.DataFrame(new_data)
    
    print("Predicting for the following new bookings:")
    print(new_bookings_df)
    
    # Use the trained pipeline to predict
    predictions = model_pipeline.predict(new_bookings_df)
    
    # Get prediction probabilities (e.g., [0.1, 0.9] means 10% low, 90% high)
    probabilities = model_pipeline.predict_proba(new_bookings_df)
    
    print("\n--- Predictions ---")
    for i, row in new_bookings_df.iterrows():
        pred_class = predictions[i]
        pred_prob = probabilities[i][pred_class]
        
        demand_label = "High Demand" if pred_class == 1 else "Low Demand"
        
        print(f"\nBooking {i+1} (Location: {row['location']}, Type: {row['customer_type']}, ADR: {row['adr']}):")
        print(f"  -> Prediction: {demand_label} ({pred_class})")
        print(f"  -> Confidence: {pred_prob * 100:.2f}%")

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Generate the data
    hotel_data = create_synthetic_data(num_samples=2000)
    
    # 2. Train the model
    # We pass the full dataframe, and the function handles splitting and preprocessing
    trained_model = train_demand_predictor(hotel_data)
    
    # 3. Use the model for prediction
    predict_new_bookings(trained_model)