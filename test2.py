
"""
HOTEL BOOKING DEMAND CLASSIFICATION AI MODEL
Target: Predict high_demand (high-demand segment classification)
Baseline Accuracy to Beat: 61.71%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class HotelDemandClassifier:
    """
    AI model to predict hotel booking demand classification
    """

    def __init__(self):
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

    def preprocess_data(self, df, is_training=True):
        """Preprocess the hotel booking data"""
        X = df.copy()

        # Remove ID and noise features
        drop_cols = ['booking_id', 'random_noise1', 'random_noise2']
        drop_cols = [c for c in drop_cols if c in X.columns]
        X = X.drop(drop_cols, axis=1)

        # Identify feature types
        numeric_features = ['lead_time_days', 'people_count', 'avg_room_rate', 
                          'nights_stayed', 'has_breakfast_included', 
                          'rating_given', 'loyalty_member']
        categorical_features = ['room_type', 'booking_channel', 'customer_country']

        # Handle missing values
        if is_training:
            X[numeric_features] = self.imputer_num.fit_transform(X[numeric_features])
            X[categorical_features] = self.imputer_cat.fit_transform(X[categorical_features])
        else:
            X[numeric_features] = self.imputer_num.transform(X[numeric_features])
            X[categorical_features] = self.imputer_cat.transform(X[categorical_features])

        # Feature Engineering
        X['is_last_minute'] = (X['lead_time_days'] <= 7).astype(int)
        X['is_advanced_booking'] = (X['lead_time_days'] >= 90).astype(int)
        X['room_rate_category'] = pd.cut(X['avg_room_rate'], 
                                         bins=[0, 5000, 10000, 15000],
                                         labels=['low', 'medium', 'high'])
        X['total_booking_value'] = X['avg_room_rate'] * X['nights_stayed']
        X['is_group'] = (X['people_count'] >= 4).astype(int)
        X['is_short_stay'] = (X['nights_stayed'] <= 2).astype(int)
        X['is_long_stay'] = (X['nights_stayed'] >= 7).astype(int)
        X['high_rating'] = (X['rating_given'] >= 4.0).astype(int)
        X['low_rating'] = (X['rating_given'] <= 2.5).astype(int)
        X['is_premium'] = ((X['room_type'] == 'Suite') | (X['avg_room_rate'] > 12000)).astype(int)
        
        # Additional feature engineering
        X['rate_per_night'] = X['avg_room_rate'] / (X['nights_stayed'] + 1)
        X['booking_value_per_person'] = X['total_booking_value'] / (X['people_count'] + 1)
        X['lead_time_category'] = pd.cut(X['lead_time_days'], 
                                          bins=[-1, 7, 30, 90, 400],
                                          labels=['immediate', 'short', 'medium', 'long'])
        X['people_nights_interaction'] = X['people_count'] * X['nights_stayed']
        X['premium_loyalty'] = X['is_premium'] * X['loyalty_member']

        # Encode categorical variables
        categorical_features.append('room_rate_category')
        categorical_features.append('lead_time_category')

        if is_training:
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_features:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        return X

    def train(self, X_train, y_train):
        """Train the model"""
        # Preprocess
        X_processed = self.preprocess_data(X_train, is_training=True)

        # Scale
        X_scaled = self.scaler.fit_transform(X_processed)

        # Create ensemble of multiple models
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=1.8,
            eval_metric='logloss'
        )
        
        # Voting ensemble with weighted voting
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
            voting='soft',
            weights=[2, 1, 2],
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y_train)
        self.feature_names = X_processed.columns

        return self

    def predict(self, X):
        """Make predictions"""
        X_processed = self.preprocess_data(X, is_training=False)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_processed = self.preprocess_data(X, is_training=False)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        return accuracy

# Example Usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Comp_Hotel_competition.csv')

    # Prepare data (replace 'high_demand' with your actual target column)
    target_col = 'strategic_family'  # The actual target column in the dataset
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    classifier = HotelDemandClassifier()
    classifier.train(X_train, y_train)

    # Evaluate
    classifier.evaluate(X_test, y_test)

    # Make predictions
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)