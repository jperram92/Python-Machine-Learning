import pandas as pd
import random
from faker import Faker
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from sklearn.model_selection import StratifiedKFold

# Step 1: Generate Fake Customer Data
fake = Faker()
Faker.seed(42)
random.seed(42)

# Predefined suburb list
PREDEFINED_SUBURBS = ['North Sydney', 'East Melbourne', 'South Brisbane', 'West Perth', 'Central Adelaide']

# Generate fake data with constrained suburbs and synthetic relationships
def generate_fake_customers(n=1000):
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

# Generate initial dataset
customer_data = generate_fake_customers(1000)
print(customer_data.head())

# Step 2: Prepare Data for Machine Learning
customer_data['dob'] = pd.to_datetime(customer_data['dob'])
customer_data['dob_year'] = customer_data['dob'].dt.year
customer_data = customer_data.drop(['dob'], axis=1)

# Encode categorical variables
label_encoders = {}
for col in ['first_name', 'last_name', 'suburb', 'gender', 'income_bracket']:
    le = LabelEncoder()
    customer_data[col] = le.fit_transform(customer_data[col])
    label_encoders[col] = le

# Define Features and Labels
X = customer_data.drop('suburb', axis=1)
y = customer_data['suburb']

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
gb_clf = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
clf = grid_search.best_estimator_

# Step 4: Alternative Model - XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train, y_train)

# Step 5: Advanced Model - CatBoost
cat_clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, verbose=0, random_state=42)
cat_clf.fit(X_train, y_train)

# Step 6: Optimized Stacking Classifier
estimators = [
    ('cat', cat_clf),
    ('gb', clf)
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42),
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)

# Evaluate Models on Validation Set
print("Gradient Boosting Validation Accuracy:", accuracy_score(y_valid, clf.predict(X_valid)))
print("XGBoost Validation Accuracy:", accuracy_score(y_valid, xgb_clf.predict(X_valid)))
print("CatBoost Validation Accuracy:", accuracy_score(y_valid, cat_clf.predict(X_valid)))
print("Stacking Classifier Validation Accuracy:", accuracy_score(y_valid, stacking_clf.predict(X_valid)))

# Cross-Validation Scores
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    xgb_clf.fit(X_train_cv, y_train_cv)
    accuracy = accuracy_score(y_test_cv, xgb_clf.predict(X_test_cv))
    cv_scores.append(accuracy)

print("XGBoost Cross-Validation Accuracy:", sum(cv_scores) / len(cv_scores))

# Step 7: Predict Next 1000 Customers with XGBoost
def predict_new_customers(n=1000, model=xgb_clf):
    new_customers = []
    for _ in range(n):
        new_customer = {
            'first_name': random.choice(customer_data['first_name'].unique()),
            'last_name': random.choice(customer_data['last_name'].unique()),
            'gender': random.choice(customer_data['gender'].unique()),
            'income_bracket': random.choice(customer_data['income_bracket'].unique()),
            'dob_year': random.randint(1930, 2005)
        }
        new_customers.append(new_customer)
    new_customers_df = pd.DataFrame(new_customers)
    predicted_suburbs = model.predict(new_customers_df)
    new_customers_df['suburb'] = label_encoders['suburb'].inverse_transform(predicted_suburbs)
    return new_customers_df

# Generate and display new customers
new_customers = predict_new_customers(model=xgb_clf)
print(new_customers.head())

# Save the best model
joblib.dump(xgb_clf, 'xgboost_best_model.pkl')

# Additional Insights
print("Feature Importances:")
importances = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_clf.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
print(importances)

# Visualize Feature Importances
plt.figure(figsize=(12, 8))
plt.barh(importances['Feature'], importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances from XGBoost Classifier')
plt.gca().invert_yaxis()
plt.show()