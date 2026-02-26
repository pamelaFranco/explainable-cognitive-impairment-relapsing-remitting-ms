###############################################################################
# This code was developed by Dr. Pamela Franco for a project focused on 
# multiple scleorosis using BMTDELAY cognitive test. 
#
# A machine learning pipeline was designed, incorporating 
# Sequential Feature Selection (SFS) for feature selection. The regression 
# models used include Random Forest, XGBoost, CatBoost, Ridge, Lasso, Logistic
# Regression, Gradient Boosting, LightGBM, ElasticNet, K-Nearest Neighbors, 
# Decision Tree, Bayesian Ridge, MLP Regressor, Quantile Regression, AdaBoost,
# and SVM with an RBF kernel. The model evaluation is performed using 5-fold 
# cross-validation with an 80%-20% train-test split. Calculated MSE, MAE, 
# and R-squared. Visualizations include fo hierarcuical clustering, regression 
# plot, and residual plots. Explainability is evaluated using SHAP (for
# tree-based models) or LIME (for other models).
#
# This code performance the stability analysis of the ML approach
#
#   Author:      Dr. Pamela Franco
#   Time-stamp:  2024-12-19
#   E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################

# SFS as feature selection and Random Forest classifier
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np 
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.inspection import DecisionBoundaryDisplay

# Plot settings
plt.rcParams['text.usetex'] = True  
plt.rcParams['font.family'] = 'serif'

###############################################################################
# 1. Load data
###############################################################################
data_file = 'MS_FA_labels_BMTDELAY.csv'
df_raw = pd.read_csv(data_file)
features = df_raw.drop(['Label'], axis=1)
labels = df_raw['Label'].values
classes = ['HC_CP', 'RRMS_CP', 'RRMS_CI']

df = features.rename(columns=lambda x: re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df.columns)

###############################################################################
# 2. Hyperparameter Optimization
###############################################################################
st_kfold = StratifiedKFold(n_splits=5, random_state=92, shuffle=True)
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20),
    'max_leaf_nodes': randint(2, 10),
    'min_samples_split': randint(5, 50)
}

rf = RandomForestClassifier(random_state=42)
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                 n_iter=10, cv=st_kfold, random_state=42)
rand_search.fit(df.values, labels)
best_rf = rand_search.best_estimator_

###############################################################################
# 3. FEATURE SELECTION STABILITY ANALYSIS WITH PROGRESS TRACKING
###############################################################################
print("\n" + "="*50)
print("STARTING FEATURE STABILITY ANALYSIS")
print("="*50)

n_iterations = 50 
stability_results = []

for i in range(n_iterations):
    print(f"\n>>> PROCESSING STABILITY ITERATION {i+1}/{n_iterations} (Random Seed: {i})")
    stability_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    
    # verbose=2 will show progress of feature selection within each iteration
    sfs_stability = SFS(best_rf, 
                        k_features=20, 
                        forward=True, 
                        floating=False, 
                        scoring='accuracy',
                        cv=stability_cv,
                        n_jobs=-1,
                        verbose=2) # <--- THIS ENABLES INTERNAL TRACKING
    
    sfs_stability.fit(df.values, labels)
    
    selected_indices = list(sfs_stability.k_feature_idx_)
    selected_names = [feature_names[idx] for idx in selected_indices]
    stability_results.extend(selected_names)
    print(f"DONE: Iteration {i+1} completed.")

selection_frequency = pd.Series(stability_results).value_counts()
stability_score = selection_frequency / n_iterations 

# Plot Feature Stability
plt.figure(figsize=(10, 8))
stability_score.head(20).plot(kind='barh', color='skyblue', edgecolor='navy')
plt.title('Feature Selection Stability Analysis (Top 20 Features)')
plt.xlabel('Selection Frequency (Stability Index)')
plt.ylabel('U-Fiber Regions')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

top_stable_features = stability_score.index[:2].tolist()

###############################################################################
# 4. Standard SFS (Final Performance Curve)
###############################################################################
print("\n" + "="*50)
print("RUNNING FINAL SFS FOR PERFORMANCE LANDSCAPE")
print("="*50)

sfs_final = SFS(best_rf, k_features=(1, 50), forward=True, floating=False, 
                scoring='accuracy', cv=st_kfold, n_jobs=-1, verbose=2)
sfs_final.fit(df.values, labels)

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs_final.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (Performance Landscape)')
plt.grid()
plt.show()

###############################################################################
# 5. Decision Boundary Display
###############################################################################
# (Mantenemos el resto del código igual para generar los límites de decisión)
plt.figure(figsize=(15, 10))
X_selected = df[top_stable_features].values
plot_colors = "ryb"
feature_pairs = list(itertools.combinations(range(len(top_stable_features)), 2))[:6]

for pairidx, pair in enumerate(feature_pairs):
    X_pair = X_selected[:, pair]
    clf = RandomForestClassifier(
        n_estimators=rand_search.best_params_["n_estimators"],
        max_depth=rand_search.best_params_["max_depth"],
        min_samples_split=rand_search.best_params_["min_samples_split"],
        max_leaf_nodes=rand_search.best_params_["max_leaf_nodes"],
        random_state=92
    ).fit(X_pair, labels)

    ax = plt.subplot(2, 3, pairidx + 1)
    DecisionBoundaryDisplay.from_estimator(
        clf, X_pair, cmap=plt.cm.RdYlBu, response_method="predict",
        ax=ax, xlabel=top_stable_features[pair[0]], ylabel=top_stable_features[pair[1]],
        alpha=0.8,
    )

    for i, color in zip(range(len(classes)), plot_colors):
        idx = np.where(labels == i)
        ax.scatter(X_pair[idx, 0], X_pair[idx, 1], c=color, label=classes[i],
                   edgecolor="black", s=50)

plt.suptitle("Decision Boundaries based on the Most Stable U-Fiber Features")
plt.tight_layout()
plt.show()