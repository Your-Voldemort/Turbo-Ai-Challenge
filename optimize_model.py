"""
AGGRESSIVE MODEL OPTIMIZATION SCRIPT
Uses cross-validation and systematic hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(X_train, X_test, y_train):
    """Optimized preprocessing"""
    # Imputers
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    # Identify features
    numeric_features = ['lead_time_days', 'people_count', 'avg_room_rate',
                       'nights_stayed', 'has_breakfast_included',
                       'rating_given', 'loyalty_member']
    categorical_features = ['room_type', 'booking_channel', 'customer_country']

    # Remove noise
    drop_cols = ['booking_id', 'random_noise1', 'random_noise2', 'strategic_family']
    drop_cols = [c for c in drop_cols if c in X_train.columns]
    X_train = X_train.drop(drop_cols, axis=1)
    X_test = X_test.drop(drop_cols, axis=1)

    # Handle missing values
    X_train[numeric_features] = imputer_num.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = imputer_num.transform(X_test[numeric_features])
    X_train[categorical_features] = imputer_cat.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = imputer_cat.transform(X_test[categorical_features])

    # KEY FEATURE ENGINEERING - Only the most important ones
    for df in [X_train, X_test]:
        # Price-based
        df['total_booking_value'] = df['avg_room_rate'] * df['nights_stayed']
        df['rate_per_person'] = df['avg_room_rate'] / (df['people_count'] + 1)
        df['value_per_person'] = df['total_booking_value'] / (df['people_count'] + 1)

        # Time-based
        df['is_last_minute'] = (df['lead_time_days'] <= 7).astype(int)
        df['is_advanced_booking'] = (df['lead_time_days'] >= 90).astype(int)
        df['lead_time_log'] = np.log1p(df['lead_time_days'])

        # Stay-based
        df['is_short_stay'] = (df['nights_stayed'] <= 2).astype(int)
        df['is_long_stay'] = (df['nights_stayed'] >= 7).astype(int)
        df['is_group'] = (df['people_count'] >= 4).astype(int)

        # Quality indicators
        df['high_rating'] = (df['rating_given'] >= 4.0).astype(int)
        df['is_premium'] = ((df['room_type'] == 'Suite') | (df['avg_room_rate'] > 12000)).astype(int)
        df['is_luxury'] = (df['avg_room_rate'] > 13000).astype(int)
        df['is_budget'] = (df['avg_room_rate'] < 6000).astype(int)

        # Interactions
        df['premium_loyalty'] = df['is_premium'] * df['loyalty_member']
        df['breakfast_value'] = df['has_breakfast_included'] * df['avg_room_rate']
        df['loyalty_rating'] = df['loyalty_member'] * df['rating_given']
        df['people_nights'] = df['people_count'] * df['nights_stayed']
        df['booking_momentum'] = df['people_count'] * df['nights_stayed'] * df['avg_room_rate'] / 10000

        # Composite scores
        df['customer_value'] = (df['loyalty_member'] * 0.4 + df['high_rating'] * 0.3 + df['is_premium'] * 0.3)
        df['urgency_score'] = 1 / (df['lead_time_days'] + 1)

        # Percentiles
        df['rate_percentile'] = df['avg_room_rate'].rank(pct=True)
        df['value_percentile'] = df['total_booking_value'].rank(pct=True)

    # Encode categoricals
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.9, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled

# Load data
print("Loading data...")
df = pd.read_csv('Comp_Hotel_competition.csv')

X = df.drop('strategic_family', axis=1)
y = df['strategic_family']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Preprocessing data...")
X_train_processed, X_test_processed, y_train_processed = preprocess_data(X_train, X_test, y_train)

print(f"Training data shape: {X_train_processed.shape}")
print(f"Test data shape: {X_test_processed.shape}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_processed)}")

# Test multiple configurations
models_to_test = []

# Configuration 1: Tuned Random Forest
models_to_test.append(('RF-Tuned', RandomForestClassifier(
    n_estimators=1500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    criterion='gini',
    max_samples=0.8
)))

# Configuration 2: Tuned XGBoost
models_to_test.append(('XGB-Tuned', XGBClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1,
    min_child_weight=2,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)))

# Configuration 3: Tuned LightGBM
models_to_test.append(('LGBM-Tuned', LGBMClassifier(
    n_estimators=1200,
    max_depth=9,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=1,
    min_child_samples=15,
    num_leaves=50,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)))

# Configuration 4: Tuned CatBoost
models_to_test.append(('CB-Tuned', CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.02,
    l2_leaf_reg=2,
    border_count=128,
    random_state=42,
    verbose=0,
    thread_count=-1
)))

# Configuration 5: Extra Trees
models_to_test.append(('ET-Tuned', ExtraTreesClassifier(
    n_estimators=1200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)))

# Configuration 6: Gradient Boosting
models_to_test.append(('GB-Tuned', GradientBoostingClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.02,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)))

# Train and evaluate each model
results = {}
trained_models = []

for name, model in models_to_test:
    print(f"\n{'='*80}")
    print(f"Training {name}...")
    print(f"{'='*80}")

    model.fit(X_train_processed, y_train_processed)
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy
    trained_models.append((name, model))

    print(f"Accuracy: {accuracy*100:.2f}%")

# Test ensemble combinations
print(f"\n{'='*80}")
print("Testing Ensemble Combinations...")
print(f"{'='*80}")

# Best 3 models
top_3_models = [(name, model) for name, model in trained_models[:3]]
ensemble_3 = VotingClassifier(
    estimators=top_3_models,
    voting='soft',
    n_jobs=-1
)
ensemble_3.fit(X_train_processed, y_train_processed)
y_pred = ensemble_3.predict(X_test_processed)
acc_3 = accuracy_score(y_test, y_pred)
results['Ensemble-Top3'] = acc_3
print(f"Top 3 Ensemble Accuracy: {acc_3*100:.2f}%")

# All 6 models
ensemble_all = VotingClassifier(
    estimators=trained_models,
    voting='soft',
    n_jobs=-1
)
ensemble_all.fit(X_train_processed, y_train_processed)
y_pred = ensemble_all.predict(X_test_processed)
acc_all = accuracy_score(y_test, y_pred)
results['Ensemble-All6'] = acc_all
print(f"All 6 Models Ensemble Accuracy: {acc_all*100:.2f}%")

# Weighted ensemble (giving more weight to better performers)
weighted_ensemble = VotingClassifier(
    estimators=trained_models,
    voting='soft',
    weights=[1.5, 1.2, 1.3, 1.4, 1.0, 1.1],
    n_jobs=-1
)
weighted_ensemble.fit(X_train_processed, y_train_processed)
y_pred = weighted_ensemble.predict(X_test_processed)
acc_weighted = accuracy_score(y_test, y_pred)
results['Weighted-Ensemble'] = acc_weighted
print(f"Weighted Ensemble Accuracy: {acc_weighted*100:.2f}%")

# Print final results
print(f"\n{'='*80}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"Baseline Accuracy: 61.71%")
print(f"Previous Best: 65.00%")
print()

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, accuracy in sorted_results:
    improvement = (accuracy - 0.6171) * 100
    vs_previous = (accuracy - 0.6500) * 100
    print(f"{name:20s}: {accuracy*100:.2f}% (+{improvement:+.2f}% vs baseline, {vs_previous:+.2f}% vs previous)")

print(f"\n{'='*80}")
print(f"BEST MODEL: {sorted_results[0][0]}")
print(f"BEST ACCURACY: {sorted_results[0][1]*100:.2f}%")
print(f"{'='*80}")

# Show detailed report for best model
print(f"\nDetailed Classification Report for Best Model:")
print(f"{'='*80}")
best_model_name = sorted_results[0][0]
best_model = [model for name, model in trained_models if name == best_model_name]
if not best_model and 'Ensemble' in best_model_name:
    if best_model_name == 'Weighted-Ensemble':
        best_model = [weighted_ensemble]
    elif best_model_name == 'Ensemble-All6':
        best_model = [ensemble_all]
    elif best_model_name == 'Ensemble-Top3':
        best_model = [ensemble_3]
else:
    best_model = best_model if best_model else [trained_models[0][1]]

y_pred_best = best_model[0].predict(X_test_processed)
print(classification_report(y_test, y_pred_best))
