import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
expr = pd.read_csv("/content/drive/MyDrive/Projects/AML/DE_genes_AML.csv", index_col=0)
labels = pd.read_csv("/content/drive/MyDrive/Projects/AML/metadata.csv", index_col=0)
# Ensure order matches between expression and labels
expr_vt = expr.T
expr_vt = expr_t.loc[labels.index]
y = labels['Condition'].map({'CONTROL':0, 'CASE':1})

# quick checks
print("expr shape:", expr.shape)
print("After transpose:", expr_vt.shape)
print("labels distribution:\n", y.value_counts())



print(expr_vt.shape)
print(y.shape)
print(expr_vt.index[:5])
print(y.index[:5])

# ANOVA F-test here to pick top-K genes (K tuned based on sample size)
K = int(expr_vt.shape[1] * 0.15) 
skb = SelectKBest(score_func=f_classif, k=min(K, expr_vt.shape[1]))
X_pre = pd.DataFrame(skb.fit_transform(expr_vt, y), index=expr_vt.index,
                     columns=expr_vt.columns[skb.get_support()])
print("After SelectKBest:", X_pre.shape)

from sklearn.preprocessing import MinMaxScaler

# Normalize features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_pre)

# Reconstructs DataFrame with original feature names
X_scaled_df = pd.DataFrame(X_scaled, index=X_pre.index, columns=X_pre.columns)

# 4) Train/test split (hold-out for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, stratify=y, test_size=0.2, random_state=42)

# 5) Handle imbalance: using scale_pos_weight in XGBoost
# compute scale_pos_weight = n_neg / n_pos
n_pos = sum(y_train==1)
n_neg = sum(y_train==0)
scale_pos_weight = n_neg / max(1, n_pos)
print("scale_pos_weight:", scale_pos_weight)

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    tree_method='gpu_hist',   # <-- enables GPU
    predictor='gpu_predictor',# <-- use GPU for prediction too
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# 7) Hyperparameter search (RandomizedSearchCV)
param_dist = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [3, 4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'reg_alpha': [0, 1e-2, 0.1, 1.0],
    'reg_lambda': [1.0, 2.0, 5.0]
}


from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method="gpu_hist")
print(xgb)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
rs = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=50, scoring='roc_auc',
                        n_jobs=6, cv=cv, verbose=2, random_state=42)

rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
print("Best CV AUC:", rs.best_score_)

best_model = rs.best_estimator_

# Final evaluation on hold-out
y_proba = best_model.predict_proba(X_test)[:,1]
y_pred = best_model.predict(X_test)
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

#feature importance hierarchy
feature_names = expr_vt.columns[skb.get_support()]
X_train_df = pd.DataFrame(X_train, columns=feature_names)

feat_imp = pd.Series(best_model.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)
print(feat_imp.head(30))

# Feature importance (gain) and SHAP interpretation
# feature importances
feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feat_imp.head(30))

# SHAP summary
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")  # global importance
shap.summary_plot(shap_values, X_train)  # beeswarm

# Save model & selected features
joblib.dump(best_model, "xgb_leukemia_model.joblib")
feat_imp.head(200).to_csv("top_features.csv")

# ROC curve plot
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - XGBoost on Leukemia')
plt.legend()
plt.savefig('roc_xgb.png', dpi=150)
