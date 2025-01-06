# ===========================================
# 📊 Fake Customer Data Analysis Pipeline
# ===========================================

# 🛠️ STEP 1: Import Required Libraries
import pandas as pd
import random
from faker import Faker
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import joblib

# ===========================================
# 🛠️ STEP 2: Generate Fake Customer Data
# ===========================================

fake = Faker()
Faker.seed(42)
random.seed(42)

# Predefined suburb list
PREDEFINED_SUBURBS = ['North Sydney', 'East Melbourne', 'South Brisbane', 'West Perth', 'Central Adelaide']

def generate_fake_customers(n=1000):
    """
    Generate fake customer data with synthetic relationships.
    """
    data = []
    for _ in range(n):
        gender = random.choice(['Male', 'Female'])
        income_bracket = random.choice(['Low', 'Medium', 'High'])
        suburb = random.choice(PREDEFINED_SUBURBS)
        if gender == 'Male' and income_bracket == 'High':
            suburb = 'North Sydney'
        elif gender == 'Female' and income_bracket == 'Low':
            suburb = 'South Brisbane'
        data.append({
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'dob': fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
            'gender': gender,
            'income_bracket': income_bracket,
            'suburb': suburb
        })
    return pd.DataFrame(data)

# Generate Dataset
customer_data = generate_fake_customers(1000)
print(customer_data.head())

# ===========================================
# 📊 STEP 3: Prepare Data for Analysis
# ===========================================

# Feature Engineering and Preprocessing
def prepare_data(df):
    """
    Prepare data for machine learning, including encoding and scaling.
    """
    df['dob'] = pd.to_datetime(df['dob'])
    df['dob_year'] = df['dob'].dt.year
    df.drop(['dob'], axis=1, inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in ['first_name', 'last_name', 'suburb', 'gender', 'income_bracket']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define Features and Labels
    X = df.drop('suburb', axis=1)
    y = df['suburb']

    # Feature Scaling
    scaler = StandardScaler()
    X[['dob_year']] = scaler.fit_transform(X[['dob_year']])

    # Feature Engineering
    X['gender_income'] = X['gender'] * X['income_bracket']

    return X, y, label_encoders

X, y, label_encoders = prepare_data(customer_data)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================================
# 🧠 STEP 4: Build and Train Models
# ===========================================

def train_models(X_train, y_train):
    """
    Train multiple machine learning models and optimize them.
    """
    # Hyperparameter Tuning with HalvingRandomSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gb_clf = GradientBoostingClassifier(random_state=42)
    halving_search = HalvingRandomSearchCV(estimator=gb_clf, param_distributions=param_grid,
                                           factor=2, random_state=42, scoring='accuracy', verbose=2)
    halving_search.fit(X_train, y_train)
    best_gb_clf = halving_search.best_estimator_

    # XGBoost Model
    xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_clf.fit(X_train, y_train)

    # CatBoost Model
    cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, verbose=0, random_state=42)
    cat_clf.fit(X_train, y_train)

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[('cat', cat_clf), ('gb', best_gb_clf)],
        final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
        n_jobs=-1
    )
    stacking_clf.fit(X_train, y_train)

    return best_gb_clf, xgb_clf, cat_clf, stacking_clf

gb_clf, xgb_clf, cat_clf, stacking_clf = train_models(X_train, y_train)

# ===========================================
# ⚙️ STEP 5: Evaluate Models
# ===========================================

def evaluate_models(models, X_valid, y_valid):
    """
    Evaluate models and display results.
    """
    model_results = {}
    for name, model in models.items():
        accuracy = accuracy_score(y_valid, model.predict(X_valid))
        model_results[name] = accuracy

    print(pd.DataFrame(model_results.items(), columns=['Model', 'Validation Accuracy']))

    print("\nXGBoost Classification Report:")
    print(classification_report(y_valid, models['XGBoost'].predict(X_valid)))
    cm = confusion_matrix(y_valid, models['XGBoost'].predict(X_valid))
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title('Confusion Matrix - XGBoost')
    plt.show()

models = {
    'Gradient Boosting': gb_clf,
    'XGBoost': xgb_clf,
    'CatBoost': cat_clf,
    'Stacking Classifier': stacking_clf
}

evaluate_models(models, X_valid, y_valid)

# ===========================================
# 🧠 STEP 6: Explain Predictions with SHAP
# ===========================================

# SHAP Model Explainability
explainer = shap.Explainer(xgb_clf, X_train, feature_names=X.columns)
shap_values = explainer(X_valid)

# Ensure valid DataFrame for SHAP
X_valid_df = pd.DataFrame(X_valid, columns=X.columns)

# Generate SHAP Summary Plot
shap.summary_plot(shap_values, X_valid_df, plot_type='dot')

# ===========================================
# 💾 STEP 7: Save Models
# ===========================================

joblib.dump(gb_clf, 'gradient_boosting_model.pkl')
joblib.dump(xgb_clf, 'xgboost_model.pkl')
joblib.dump(cat_clf, 'catboost_model.pkl')
joblib.dump(stacking_clf, 'stacking_model.pkl')

# ===========================================
# 📊 STEP 8: Visualize Feature Importance
# ===========================================

importances = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_clf.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
plt.barh(importances['Feature'], importances['Importance'])
for index, value in enumerate(importances['Importance']):
    plt.text(value, index, f"{value:.2f}")
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances from XGBoost Classifier')
plt.gca().invert_yaxis()
plt.show()
