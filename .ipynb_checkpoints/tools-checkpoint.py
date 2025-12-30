import subprocess
import sys
import sklearn
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from xgboost import DMatrix, train as xgb_train
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import approx_fprime



def install_and_import(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)


# List of equivalent Python packages
python_packages = [
    "plotnine",  # ggplot2 equivalent
    "pandas",  # part of tidyverse equivalent
    "matplotlib", "seaborn",  # part of cowplot, ggpubr, ggsci equivalents
    # "scikit-learn", # part of glmnet, e1071, caret, class equivalents
    "xgboost",  # direct equivalent
    "numpy", "scipy"

]

# Loop through the list and apply the function
for pkg in python_packages:
    install_and_import(pkg)



def LOGIS(train_data, train_labels, test_data, test_labels):
    r"""This is an Ridge regression classifier.
    
    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data

    """
    model = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='liblinear', scoring='accuracy', random_state=0,
                                 max_iter=1000)

    # Fit the model
    model.fit(train_data, train_labels)

    # Predict probabilities. The returned estimates for all classes are ordered by the label of classes.
    predictions_proba = model.predict_proba(test_data)

    # Convert probabilities to binary predictions using 0.5 as the threshold.
    # predictions = (predictions_proba > 0.5).astype(int)
    predictions = model.predict(test_data)

    # Calculate AUC for binary classification / multi-classification
    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(test_labels, predictions_proba, multi_class='ovo', average='macro')

    return {
        'f1': f1_score(test_labels, predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'auc': auc
    }
    

def SVM(train_data, train_labels, test_data, test_labels):
    r"""This is a Support Vector Machine classifier.

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    model = SVC(probability=True)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)
    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(test_labels, predictions_proba, multi_class='ovo', average='macro')

    return {
        'f1': f1_score(test_labels, predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'auc': auc
    }


def KNN(train_data, train_labels, test_data, test_labels):
    r"""This is a K-Nearest Neighbor classifier.

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_data, train_labels)

    # Predict the class labels for the provided data
    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    # Predict class probabilities for the positive class
    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1]) # Binary classification, get probabilities for the positive class
    else:
        auc = roc_auc_score(test_labels, predictions_proba, multi_class='ovo', average='macro')

    return {
        'f1': f1_score(test_labels, predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'auc': auc
    }


def RF(train_data, train_labels, test_data, test_labels):
    r"""This is a Random Forest classifier.

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_data, train_labels)

    predictions_proba = model.predict_proba(test_data)
    predictions = model.predict(test_data)
    if predictions_proba.shape[1] == 2:
        auc = roc_auc_score(test_labels, predictions_proba[:, 1])
    else:
        auc = roc_auc_score(test_labels, predictions_proba, multi_class='ovo', average='macro')

    return {
        'f1': f1_score(test_labels, predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'auc': auc
    }


def XGB(train_data, train_labels, test_data, test_labels):
    r"""This is an XGBoost classifier. 

    Parameters
    -----------
    train_data : pd.DataFrame
            the training data
    train_labels : pd.DataFrame
            the labels of the training data
    test_data : pd.DataFrame
            the test data
    test_labels : pd.DataFrame
            the labels of the test data
    
    """
    num_class = len(np.unique(train_labels))
    dtrain = DMatrix(train_data, label=train_labels)
    dtest = DMatrix(test_data, label=test_labels)
    # Parameters and model training
    if num_class == 2:
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
    else:
        params = {'objective': 'multi:softprob', 'num_class': num_class, 'eval_metric': 'mlogloss'}
    bst = xgb_train(params, dtrain, num_boost_round=10)
    predictions_proba = bst.predict(dtest)
    if predictions_proba.ndim == 1:
        predictions = (predictions_proba > 0.5).astype(int)
        auc = roc_auc_score(test_labels, predictions_proba)
    else:
        predictions = np.argmax(predictions_proba, axis=1)
        auc = roc_auc_score(test_labels, predictions_proba, multi_class='ovo', average='macro')
        
    return {
        'f1': f1_score(test_labels, predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'auc': auc
    }


# Assuming LOGIS, SVM, KNN, RF, and XGB functions are defined as previously discussed

def eval_classifier(whole_generated, whole_groups, n_candidate, n_draw=5, log=True, methods=None):
    r"""
    For each classifier and each candidate sample size, this function performs n_draw rounds of 
    stratified sampling from the data (proportional to class distribution), applies 5-fold cross-validation, 
    and averages metrics across draws. Used to support IPLF fitting.

    Parameters
    ----------
    whole_generated : pd.DataFrame
        The dataset to sample from.
    whole_groups : array-like
        Class labels corresponding to the dataset.
    n_candidate : list
        List of candidate sample sizes to evaluate.
    n_draw : int, default=5
        Number of resampling repetitions for each sample size.
    log : bool, default=True
        Whether the input data is already log-transformed.
    methods : list of str, optional
        List of classifier names to evaluate. Defaults to ['LOGIS', 'SVM', 'KNN', 'RF', 'XGB'].

    Returns
    -------
    pd.DataFrame
        A dataframe summarizing metrics (f1_score, accuracy, auc) across settings.
    """
    if methods is None:
        methods = ['LOGIS', 'SVM', 'KNN', 'RF', 'XGB']

    if not log:
        whole_generated = np.log2(whole_generated + 1)

    whole_groups = np.array([str(item) for item in whole_groups])
    unique_groups = np.unique(whole_groups)
    group_dict = {g: i for i, g in enumerate(unique_groups)}
    whole_labels = np.array([group_dict[g] for g in whole_groups])

    group_counts = {g: sum(whole_groups == g) for g in unique_groups}
    total = sum(group_counts.values())
    group_proportions = {g: group_counts[g] / total for g in unique_groups}
    group_indices_dict = {g: np.where(whole_groups == g)[0] for g in unique_groups}

    results = []
    classifier_map = {
        'LOGIS': LOGIS,
        'SVM': SVM,
        'KNN': KNN,
        'RF': RF,
        'XGB': XGB
    }

    for n_index, n in enumerate(n_candidate):
        print(f"\nRunning sample size index {n_index + 1}/{len(n_candidate)} (n = {n})\n")
        for draw in range(n_draw):
            indices = []
            for g in unique_groups:
                n_g = int(round(n * group_proportions[g]))
                selected = np.random.choice(group_indices_dict[g], n_g, replace=False)
                indices.extend(selected)
            indices = np.array(indices)

            dat_candidate = whole_generated.iloc[indices].values
            labels_candidate = whole_labels[indices]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # store scores for each classifier
            metrics = {
                method: {'f1': [], 'accuracy': [], 'auc': []} for method in methods
            }

            for train_index, test_index in skf.split(dat_candidate, labels_candidate):
                train_data, test_data = dat_candidate[train_index], dat_candidate[test_index]
                train_labels, test_labels = labels_candidate[train_index], labels_candidate[test_index]

                non_zero_std = train_data.std(axis=0) != 0
                train_data[:, non_zero_std] = scale(train_data[:, non_zero_std])
                test_data[:, non_zero_std] = scale(test_data[:, non_zero_std])

                for method in methods:
                    clf_func = classifier_map[method]
                    res = clf_func(train_data, train_labels, test_data, test_labels)
                    metrics[method]['f1'].append(res['f1'])
                    metrics[method]['accuracy'].append(res['accuracy'])
                    metrics[method]['auc'].append(res['auc'])

            for method in methods:
                mean_f1 = np.mean(metrics[method]['f1'])
                mean_acc = np.mean(metrics[method]['accuracy'])
                mean_auc = np.mean(metrics[method]['auc'])
                print(f"[n={n}, draw={draw}, method={method}] F1: {mean_f1:.4f}, Acc: {mean_acc:.4f}, AUC: {mean_auc:.4f}")
                results.append({
                    'total_size': n,
                    'draw': draw,
                    'method': method,
                    'f1_score': mean_f1,
                    'accuracy': mean_acc,
                    'auc': mean_auc
                })

    return pd.DataFrame(results)


def heatmap_eval(dat_real,dat_generated=None):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data.
    dat_generated is applicable only if 2 sets of data is available.

    Parameters
    -----------
    dat_real: pd.DataFrame
            the original copy of the data
    dat_generated : pd.DataFrame, optional
            the generated data
    
    """
    if dat_generated is None:
        # Only plot dat_real if dat_generated is None
        plt.figure(figsize=(6, 6))
        sns.heatmap(dat_real, cbar=True)
        plt.title('Real Data')
        plt.xlabel('Features')
        plt.ylabel('Samples')
    else:
        # Plot both dat_generated and dat_real side by side
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6),
                                gridspec_kw=dict(width_ratios=[0.5, 0.55]))

        sns.heatmap(dat_generated, ax=axs[0], cbar=False)
        axs[0].set_title('Generated Data')
        axs[0].set_xlabel('Features')
        axs[0].set_ylabel('Samples')

        sns.heatmap(dat_real, ax=axs[1], cbar=True)
        axs[1].set_title('Real Data')
        axs[1].set_xlabel('Features')
        axs[1].set_ylabel('Samples')



def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, random_state = 42, legend_pos="top"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data.
    If only 1 set of data is available, dat_generated and groups_generated should have None as inputs.

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the generated data, input None if unavailable
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series
            the groups generated, input None if unavailable
    groups_real : pd.Series
            the real groups
    legend_pos : string
            legend location
    
    """

    if dat_generated is None and groups_generated is None:
        # Only plot the real data
        reducer = UMAP(random_state=random_state)
        embedding = reducer.fit_transform(dat_real.values)

        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        umap_df['Group'] = groups_real.astype(str)  # Ensure groups are hashable for seaborn

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', style='Group', palette='bright')
        plt.legend(title='Group', loc=legend_pos)
        plt.title('UMAP Projection of Real Data')
        plt.show()
        return
    
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))  
    combined_groups = np.concatenate((groups_real, groups_generated))
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # Ensure that group labels are hashable and can be used in seaborn plots
    combined_groups = [str(group) for group in combined_groups]  # Convert groups to string if not already

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=random_state)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels
    umap_df['Group'] = combined_groups

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
    plt.legend(title='Data Type/Group', loc="best")
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()



def power_law(x, a, b, c):
    return (1 - a) - (b * (x ** c))



def fit_curve(acc_table, metric_name, n_target=None, plot=True, ax=None, annotation=("Metric", "")):
    initial_params = [0, 1, -0.5]  # Adjust based on data inspection
    max_iterations = 50000  # Increase max iterations

    popt, pcov = curve_fit(power_law, acc_table['n'], acc_table[metric_name], p0=initial_params, maxfev=max_iterations)

    acc_table['predicted'] = power_law(acc_table['n'], *popt)
    epsilon = np.sqrt(np.finfo(float).eps)
    jacobian = np.empty((len(acc_table['n']), len(popt)))
    for i, x in enumerate(acc_table['n']):
        jacobian[i] = approx_fprime([x], lambda x: power_law(x[0], *popt), epsilon)
    pred_var = np.sum((jacobian @ pcov) * jacobian, axis=1)
    pred_std = np.sqrt(pred_var)
    t = norm.ppf(0.975)
    acc_table['ci_low'] = acc_table['predicted'] - t * pred_std
    acc_table['ci_high'] = acc_table['predicted'] + t * pred_std

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(acc_table['n'], acc_table['predicted'], label='Fitted', color='blue', linestyle='--')
        ax.scatter(acc_table['n'], acc_table[metric_name], label='Actual Data', color='red')
        ax.fill_between(acc_table['n'], acc_table['ci_low'], acc_table['ci_high'], color='blue', alpha=0.2, label='95% CI')
        ax.set_xlabel('Sample Size')
        ax.legend(loc='best')
        ax.set_title(annotation)
        ax.set_ylim(0.4, 1)
      

        
        if ax is None:
            plt.show()
        return ax
    return None
    

def get_data_metrics(real_file_name, generated_file_name):
    """
    Load and preprocess real and generated datasets for downstream evaluation.

    Parameters
    ----------
    real_file_name : str
        Path to the CSV file containing real data.
    generated_file_name : str
        Path to the CSV file containing generated data.

    Returns
    -------
    real_data : pd.DataFrame
        Log2-transformed real data feature matrix.
    groups_real : pd.Series
        Binary-encoded group labels for real data (0/1).
    generated_data : pd.DataFrame
        Feature matrix for generated data (no transformation).
    groups_generated : pd.Series
        Group labels from generated data (as-is, assumed last column).
    unique_types : np.ndarray
        Array of unique binary group values (after mapping, i.e., [0, 1]).
    """
    # Load real dataset and drop non-feature column
    real = pd.read_csv(real_file_name, header=0)
    real.drop(columns='samples', inplace=True)

    # Load generated dataset and assign same column names
    generated = pd.read_csv(generated_file_name, header=None, names=real.columns)
    
    unique_types = real['groups'].unique()
    # Consistently encode the first and second group as 0 and 1
    if not np.issubdtype(real['groups'].dtype, np.number):
        type_map = {unique_types[0]: 0, unique_types[1]: 1}
        real['groups'] = real['groups'].map(type_map)
    
    # Extract group labels
    groups_real = real.groups
    groups_generated = generated.groups
    
    # Extract feature matrices
    real_data = real.iloc[:, :-1]
    real_data = np.log2(real_data + 1)  # Log-transform real data
    generated_data = generated.iloc[:, :-1]
    unique_types = real['groups'].unique() 

    # Return processed matrices and labels
    return real_data, groups_real, generated_data, groups_generated, unique_types



def visualize(real_data, groups_real, unique_types, generated_data=None, groups_generated=None, ratio=1, seed=42):
    """
    Visualize real and optionally generated data using heatmap and UMAP projections.

    Supports both binary and multi-class settings. For each class, samples from both datasets
    are drawn based on real data class proportions.

    Parameters
    ----------
    real_data : pd.DataFrame
        Feature matrix of real dataset (without 'groups' column).
    groups_real : pd.Series
        Group labels for the real dataset.
    unique_types : array-like
        Unique class labels to iterate over.
    generated_data : pd.DataFrame, optional
        Feature matrix of generated dataset (same columns as real_data).
    groups_generated : pd.Series, optional
        Group labels for the generated dataset.
    ratio : float, default=1
        Sampling ratio within each class (based on real data).
    seed : int, default=42
        Random seed for reproducibility.
    """
    np.random.seed(seed)

    real_indices = []
    generated_indices = []

    for group in unique_types:
        # Sample from real
        real_idx = np.where(groups_real == group)[0]
        n_sample = round(len(real_idx) * ratio)
        sampled_real = np.random.choice(real_idx, size=n_sample, replace=False)
        real_indices.extend(sampled_real.tolist())

        # Sample from generated if provided
        if generated_data is not None and groups_generated is not None:
            gen_idx = np.where(groups_generated == group)[0]
            if len(gen_idx) < n_sample:
                raise ValueError(f"Not enough samples in generated data for group '{group}'")
            sampled_gen = np.random.choice(gen_idx, size=n_sample, replace=False)
            generated_indices.extend(sampled_gen.tolist())

    # Heatmap
    if generated_data is None:
        heatmap_eval(dat_real=real_data.iloc[real_indices, :])
    else:
        heatmap_eval(
            dat_real=real_data.iloc[real_indices, :],
            dat_generated=generated_data.iloc[generated_indices, :]
        )

        # UMAP
        UMAP_eval(
            dat_real=real_data.iloc[real_indices, :],
            dat_generated=generated_data.iloc[generated_indices, :],
            groups_real=groups_real.iloc[real_indices],
            groups_generated=groups_generated.iloc[generated_indices],
            legend_pos="bottom"
        )




def vis_classifier(metric_real, n_target, metric_generated=None, metric_name='f1_score', save = False):
    r""" 
    Visualize the IPLF (learning curve) fitted from the real and optionally generated samples.

    Parameters
    ----------
    metric_real : pd.DataFrame
        Metrics from eval_classifier on real data (must contain column matching `metric_name`).
    n_target : int or array-like
        Sample sizes beyond candidate range for extrapolation.
    metric_generated : pd.DataFrame, optional
        Metrics from eval_classifier on generated data.
    metric_name : str, default='f1_score'
        Metric to visualize. Options: 'f1_score', 'accuracy', 'auc'.
    """
    methods = metric_real['method'].unique()
    num_methods = len(methods)
    
    # Adjust subplot layout
    cols = 2 if metric_generated is not None else 1
    fig, axs = plt.subplots(num_methods, cols, figsize=(15, 5 * num_methods))
    if num_methods == 1:
        axs = [axs] if cols == 1 else [axs]  # keep as list or 2-list

    # Helper: compute mean per sample size
    def mean_metrics(df, metric_name):
        return df.groupby(['total_size', 'method']).agg({metric_name: 'mean'}).reset_index().rename(
            columns={metric_name: metric_name, 'total_size': 'n'}
        )

    for i, method in enumerate(methods):
        print(method)
        df_real = metric_real[metric_real['method'] == method]
        mean_real = mean_metrics(df_real, metric_name)

        if metric_generated is not None:
            df_gen = metric_generated[metric_generated['method'] == method]
            mean_gen = mean_metrics(df_gen, metric_name)

        # Plot real
        ax_real = axs[i] if cols == 1 else axs[i][0]
        fit_curve(mean_real, metric_name, n_target=n_target, plot=True,
                  ax=ax_real, annotation=(metric_name, f"{method}: Real"))

        # Plot generated
        if metric_generated is not None:
            ax_gen = axs[i][1]
            fit_curve(mean_gen, metric_name, n_target=n_target, plot=True,
                      ax=ax_gen, annotation=(metric_name, f"{method}: Generated"))

    plt.tight_layout()
    # plt.show()

    if save:
        return fig
