# SFS as feature selection and Random Forest classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
#import seaborn as sns
###############################################################################
# Load data
newData = 'MS_FA_labels_BMTDELAY.csv'
df = pd.read_csv(newData)
features = df.drop(['Label'], axis=1)
label = df['Label'].values
classes = ['HC_CP','RRMS_CP','RRMS_CI']
df = features.rename(columns = lambda x:re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df.columns)
###############################################################################
# Find Hyperparameters Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np 
from sklearn.model_selection import StratifiedKFold
seed = np.random.seed(22)
st_kfold = StratifiedKFold(n_splits=5, random_state=92, shuffle=True)
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20),
              'max_leaf_nodes':randint(2,10),
              'min_samples_split': randint(5,50)}
# Create a random forest classifier
rf = RandomForestClassifier()
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.1)
# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist,
                                 n_iter=20, 
                                 cv=st_kfold)
# Fit the random search object to the data
grid_results =rand_search.fit(X_train, y_train)
# Create a variable for the best model
best_rf = rand_search.best_estimator_
# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)
# Select the best Model and Measure Performance
best_model = grid_results.best_estimator_
y_pred = best_model.predict(X_test)
y_df = pd.DataFrame(y_test)
y_df['PredictedPrice']=y_pred
y_df.head()

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_pred, y_test)
print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))

# Mean Absolute Percentage Error (MAPE)
MAPE = mean_absolute_percentage_error(y_pred, y_test)
print('Median Absolute Percentage Error (MAPE): ' + str(np.round(MAPE*100, 2)) + ' %')

###############################################################################
# SFS as feature selection
# Features importance plot
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(features, label)
importance = np.abs(ridge.coef_)
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#importance = sorted(importance, reverse=True)
fig, ax = plt.subplots(figsize=(30,15))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(len(feature_importance_df)), feature_importance_df['Importance'],
        color='Cornflowerblue')
plt.xticks(range(len(feature_importance_df)), feature_importance_df['Feature'],
           rotation=90)
plt.xlabel('Features', fontsize=30)
plt.ylabel('Feature Importance', fontsize=30)
plt.yticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-0.5,len(feature_importance_df)-0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.savefig('featureImportanceSFS_RF.pdf')
plt.show()

# Features Selection (number of features vs. precision)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
selected_features = feature_importance_df['Feature']
X_sel = features[selected_features]
rf_model = RandomForestClassifier(n_estimators=rand_search.best_params_["n_estimators"], 
                                  max_depth=rand_search.best_params_["max_depth"], 
                                  min_samples_split=rand_search.best_params_["min_samples_split"],
                                  max_leaf_nodes=rand_search.best_params_["max_leaf_nodes"],
                                  random_state=92)
sfs=SFS(rf_model,
        k_features='best', 
        forward=True, 
        floating=False, 
        verbose=2,
        scoring='accuracy',
        cv=st_kfold)

fs_svc = sfs.fit(X_sel, label)
scores = fs_svc.subsets_[len(fs_svc.k_feature_idx_)]['cv_scores']
print('Best Score:', round(np.mean(scores),4),u"\u00B1",
      round(np.std(scores),4), '%')
print('Best subset (indices):', fs_svc.k_feature_idx_)
print('Number of features selected:', len(fs_svc.k_feature_idx_))
print('Best subset (corresponding names):', fs_svc.k_feature_names_)

# Plot feature importance
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
plot_sfs(sfs.get_metric_dict(confidence_interval=0.95), kind='std_dev',
         figsize=(30, 10))
#plt.title('Sequential Forward Selection using Standard Deviation')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlim(1,100) 
plt.yticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('feature_importanceSFS_RF.pdf')
plt.show()

X_selected = pd.DataFrame(features, columns=fs_svc.k_feature_names_)

#Plot best feature combination for SFS
feature_names = list(X_sel.columns)
selected_features = sfs.k_feature_names_
ranked_features = [feature_names[i] for i in sfs.k_feature_idx_]
feature_ranking = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
feature_ranking = pd.DataFrame({'Feature': ranked_features, 'Score': sfs.k_feature_idx_})
feature_ranking = feature_ranking.sort_values(by='Score', ascending=False)

plt.figure(figsize=(12, 10))
plt.rcParams['font.size'] = 15
plt.barh(feature_ranking['Feature'], feature_ranking['Score'], color='Cornflowerblue',height=0.5)
plt.xlabel('Importance', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.gca().invert_yaxis()  
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.savefig('bestfeatureSFS_RF.pdf')
plt.show()
###############################################################################
# tSNE features selected
tsne = TSNE()
tsne_results = tsne.fit_transform(X_selected)
# Create a scatter plot of the T-SNE results
plt.rcParams["figure.figsize"] = [5.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots()
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=label, cmap='viridis')
plt.xlabel("First Dimension",fontsize=15)
plt.ylabel("Second Dimension",fontsize=15)
legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=classes, title ='Classes')
ax.add_artist(legend1)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.show()
fig.savefig('tsneFeatureSelectedSFSRF.pdf')
###############################################################################
# SHAP-values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, label, random_state=0)
cls = RandomForestClassifier(n_estimators=rand_search.best_params_["n_estimators"], 
                                  max_depth=rand_search.best_params_["max_depth"], 
                                  min_samples_split=rand_search.best_params_["min_samples_split"],
                                  max_leaf_nodes=rand_search.best_params_["max_leaf_nodes"],
                                  random_state=92)
cls.fit(X_train, y_train)

# Compute SHAP values
import shap
# compute SHAP values
explainer = shap.TreeExplainer(cls)
shap_values = explainer.shap_values(X_selected)
# SHAP Summary Plot for Specific Class
fig, ax = plt.subplots(figsize=(30, 30))
shap.summary_plot(shap_values, X_selected, plot_type="bar", 
                  class_names= classes, feature_names = X_selected.columns)
fig.savefig('shapSummaryPlotofSpecifClassSFSRF.pdf')
# SHAP Summary Plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values[1], X_selected, feature_names = X_selected.columns)
fig.savefig('shapSummaryPlotSFSRF.pdf')
# SHAP Dependece Plot
shap.dependence_plot(0, shap_values[0], X_selected, feature_names=X_selected.columns, 
                     show=False)
plt.savefig('shapDependencePlotSFSRF.pdf')
# SHAP Waterfall Plot for Class 0, 1, and 2
fig, ax = plt.subplots()
row = 0
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                     base_values=explainer.expected_value[0],
                                     data=X_test.iloc[row],  
                                     feature_names=X_test.columns.tolist()))
fig.savefig('WaterfallClass0SFSRF.pdf')
fig, ax = plt.subplots()
row = 1
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                              base_values=explainer.expected_value[0],
                                              data=X_test.iloc[row],
                                              feature_names=X_test.columns.tolist()))
fig.savefig('WaterfallClass1SFSRF.pdf')
fig, ax = plt.subplots()
row = 2
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                              base_values=explainer.expected_value[0],
                                              data=X_test.iloc[row],
                                              feature_names=X_test.columns.tolist()))
fig.savefig('WaterfallClass2SFSRF.pdf')
# SHAP Force plot
i=2
shap.force_plot(explainer.expected_value[0], shap_values[0][i], X_selected.values[i],
                feature_names = X_selected.columns,matplotlib=True, show=False)
plt.savefig('force_plotSFSRF.pdf')
###############################################################################
# Decision Tree
from sklearn.tree import plot_tree
df = X_selected.rename(columns = lambda x:re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df.columns)

tree_to_plot = cls.estimators_[0]

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=feature_names, filled=True, class_names=classes, 
          rounded=True, fontsize=15)
#plt.title("Decision Tree from Random Forest")
plt.savefig('tree.pdf')
plt.show()

###############################################################################
# Plot decision surface using Random Forest 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler

# Standardize the features for better visualization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_selected), columns=selected_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, label, test_size=0.1, random_state=92) 

# Train the best Random Forest classifier from the earlier hyperparameter search
best_rf = RandomForestClassifier(
    n_estimators=rand_search.best_params_["n_estimators"],
    max_depth=rand_search.best_params_["max_depth"],
    min_samples_split=rand_search.best_params_["min_samples_split"],
    max_leaf_nodes=rand_search.best_params_["max_leaf_nodes"],
    random_state=92
).fit(X_train, y_train)

# Plot decision surfaces for pairs of features
plot_colors = "ryb"

# Prepare the figure with a larger size for better layout
plt.figure(figsize=(18, 12))

# Iterate over pairs of selected features
feature_pairs = [(i, j) for i in range(len(selected_features)) for j in range(i+1, len(selected_features))]

for pairidx, pair in enumerate(feature_pairs[:6]):  # Limit to 6 pairs for plotting
    # Select the corresponding features
    X_pair = X_scaled.iloc[:, list(pair)].values

    # Train the classifier on the selected pair of features
    clf = RandomForestClassifier(
        n_estimators=rand_search.best_params_["n_estimators"],
        max_depth=rand_search.best_params_["max_depth"],
        min_samples_split=rand_search.best_params_["min_samples_split"],
        max_leaf_nodes=rand_search.best_params_["max_leaf_nodes"],
        random_state=92
    ).fit(X_pair, label)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_pair,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=selected_features[pair[0]],
        ylabel=selected_features[pair[1]],
        alpha=0.8,
    )

    # Plot the training points
    for i, color in zip(range(len(classes)), plot_colors):
        idx = np.where(label == i)
        ax.scatter(
            X_pair[idx, 0],
            X_pair[idx, 1],
            c=color,
            label=classes[i],
            edgecolor="black",
            s=50,  # Increase marker size
        )

    # Set limits to ensure all data is visible
    ax.set_xlim(X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5)
    ax.set_ylim(X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5)

# Adjust layout to ensure no overlap and better spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Decision Surface of Random Forest Trained on Pairs of Selected Features", fontsize=16)
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
plt.savefig('decisionsurfaceRF.pdf')
plt.show()