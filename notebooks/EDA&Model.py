import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Network_Data/phisingData.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nTarget Distribution:")
print(df['Result'].value_counts())
print(df['Result'].value_counts(normalize=True))

plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
df['Result'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Result')
plt.ylabel('Count')

plt.subplot(2, 3, 2)
feature_means = df.groupby('Result').mean()
top_features = feature_means.abs().mean().sort_values(ascending=False).head(10)
top_features.plot(kind='bar')
plt.title('Top 10 Features by Average Absolute Value')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix['Result'].sort_values(ascending=False).head(15).to_frame().T, 
            annot=True, cmap='coolwarm', center=0)
plt.title('Top 15 Features Correlation with Target')

plt.subplot(2, 3, 4)
feature_variance = df.drop('Result', axis=1).var().sort_values(ascending=False).head(10)
feature_variance.plot(kind='bar')
plt.title('Top 10 Features by Variance')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
phishing_features = df[df['Result'] == 1].drop('Result', axis=1).mean()
legitimate_features = df[df['Result'] == -1].drop('Result', axis=1).mean()
feature_diff = (phishing_features - legitimate_features).abs().sort_values(ascending=False).head(10)
feature_diff.plot(kind='bar')
plt.title('Top 10 Discriminative Features')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
feature_counts = {}
for col in df.columns[:-1]:
    unique_vals = df[col].nunique()
    feature_counts[col] = unique_vals
feature_diversity = pd.Series(feature_counts).sort_values(ascending=False).head(10)
feature_diversity.plot(kind='bar')
plt.title('Feature Value Diversity')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

selected_features = correlation_matrix['Result'].abs().sort_values(ascending=False).head(20).index[1:].tolist()
print(f"\nSelected Features: {selected_features}")

X = df[selected_features]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

baseline_results = {}
for name, model in models.items():
    if name in ['LogisticRegression', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    baseline_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}

best_models = {}
tuned_results = {}

for name, model in models.items():
    print(f"\nTuning {name}...")
    
    grid_search = GridSearchCV(
        model, param_grids[name], 
        cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=0
    )
    
    if name in ['LogisticRegression', 'SVM']:
        grid_search.fit(X_train_scaled, y_train)
        y_pred = grid_search.predict(X_test_scaled)
        y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
    else:
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    best_models[name] = grid_search.best_estimator_
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    tuned_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Tuned {name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

ensemble = VotingClassifier([
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('lr', best_models['LogisticRegression'])
], voting='soft')

ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_roc_auc = roc_auc_score(y_test, ensemble_pred_proba)

print(f"\nEnsemble Model - Accuracy: {ensemble_accuracy:.4f}, ROC-AUC: {ensemble_roc_auc:.4f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
results_df = pd.DataFrame({
    'Baseline': [baseline_results[model]['accuracy'] for model in models.keys()],
    'Tuned': [tuned_results[model]['accuracy'] for model in models.keys()]
}, index=list(models.keys()))
results_df.plot(kind='bar')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()

plt.subplot(2, 3, 2)
roc_df = pd.DataFrame({
    'Baseline': [baseline_results[model]['roc_auc'] for model in models.keys()],
    'Tuned': [tuned_results[model]['roc_auc'] for model in models.keys()]
}, index=list(models.keys()))
roc_df.plot(kind='bar')
plt.title('Model ROC-AUC Comparison')
plt.ylabel('ROC-AUC')
plt.xticks(rotation=45)
plt.legend()

plt.subplot(2, 3, 3)
best_model_name = max(tuned_results.keys(), key=lambda k: tuned_results[k]['roc_auc'])
best_model = best_models[best_model_name]

if best_model_name in ['LogisticRegression', 'SVM']:
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc_score(y_test, y_pred_proba_best):.3f})')

fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_pred_proba)
plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC = {ensemble_roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Model Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 3, 5)
if hasattr(best_models['RandomForest'], 'feature_importances_'):
    feature_importance = pd.Series(
        best_models['RandomForest'].feature_importances_, 
        index=selected_features
    ).sort_values(ascending=False).head(10)
    feature_importance.plot(kind='bar')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
cross_val_scores = {}
for name, model in best_models.items():
    if name in ['LogisticRegression', 'SVM']:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    cross_val_scores[name] = scores.mean()

pd.Series(cross_val_scores).plot(kind='bar')
plt.title('Cross-Validation ROC-AUC Scores')
plt.ylabel('Mean ROC-AUC')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*50)

for name, results in tuned_results.items():
    print(f"{name:20} - Accuracy: {results['accuracy']:.4f}, ROC-AUC: {results['roc_auc']:.4f}")

print(f"{'Ensemble':20} - Accuracy: {ensemble_accuracy:.4f}, ROC-AUC: {ensemble_roc_auc:.4f}")

print(f"\nBest Individual Model: {best_model_name}")
print(f"Best Overall Model: {'Ensemble' if ensemble_roc_auc > max(tuned_results.values(), key=lambda x: x['roc_auc'])['roc_auc'] else best_model_name}")

print("\nClassification Report for Ensemble Model:")
print(classification_report(y_test, ensemble_pred))