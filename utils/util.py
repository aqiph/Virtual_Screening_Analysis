#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 08 14:32:00 2025

@author: guohan

"""

import os, subprocess, json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_size(12)
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


### Plot property distribution ###
def plot_property_distribution(input_file, property_list=None, method='hist'):
    """
    Plot property distribution.
    :param input_file: str, path of the input file.
    :param property_list: list of property names.
    :param method: str, method of plotting, allowed values include 'hist' and kde
    """
    colors = ['b', 'r', 'orange', 'green', 'purple']
    if property_list is None:
        property_list = ['Docking_Score', 'MW', 'logP', 'TPSA']

    df = pd.read_csv(input_file)

    for i, property in enumerate(property_list):
        color = colors[i%5]
        plt.figure(figsize=(8, 5))
        if method == 'hist':
            sns.histplot(df[property], kde=False, color=color, linewidth=1, label=property)
            plt.ylabel("Count", fontsize=16)
        elif method == 'kde':
            sns.kdeplot(df[property], fill=True, color=color, linewidth=2, label=property)
            plt.ylabel("Density", fontsize=16)
        plt.xlabel(property, fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        plt.title(f'{property} Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{property}_Distribution.pdf', format='pdf')


### Plot docking score vs. MW ###
def plot_dockingScore_vs_MW(input_file, smiles_column_name='Cleaned_SMILES', dockingScore_column_name='Docking_Score'):
    """
    Plot docking score vs. MW, compute molecular weight bias in docking score.
    :param input_file: str, path of the input file.
    :param smiles_column_name: str, name of the SMILES column.
    :param dockingScore_column_name: str, name of the docking score column.
    """
    df = pd.read_csv(input_file)
    df.dropna(subset=[dockingScore_column_name], inplace=True)
    print(f'The number of rows is {df.shape[0]}')

    smiles = df[smiles_column_name].tolist()
    dockingScore = df[dockingScore_column_name].tolist()
    if 'MW' in df.columns.tolist():
        mw = df['MW'].tolist()
    else:
        mw = [round(Descriptors.MolWt(Chem.MolFromSmiles(s)), 1) for s in smiles]
    mw = np.array(mw)
    dockingScore = np.array(dockingScore)

    # calculate Pearson correlation coefficient and P-value
    r_value, p_value = pearsonr(mw, dockingScore)
    print(f"Pearson correlation coefficient: {r_value:.4f}")
    print(f"P-value: {p_value:.4e}")

    # plot
    plt.figure(figsize=(8, 5))
    plt.scatter(mw, dockingScore, color='black', s = 2, marker='o')

    slope, intercept = np.polyfit(mw, dockingScore, 1)
    print(f'The slope for the trend line is {slope:.4f}')
    trendline = slope * mw + intercept
    plt.plot(mw, trendline, color='red', linestyle='--', label='Trend Line')

    plt.xlabel("Molecular Weight", fontsize=16)
    plt.ylabel("Docking Score", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.title('Docking Score vs. Molecular Weight', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'Docking Score vs Molecular Weight.pdf', format='pdf')


### Preprocess boltz2 binding affinity files ###
def get_boltzResult(id, input_dir_boltzScore):
    """
    Extract Boltz results, helper function for preprocessing Boltz results.
    :param id: str, id of the input file.
    :param input_dir_boltzScore: str, path of the Boltz result directory.
    """
    file = f'{input_dir_boltzScore}/predictions/{id}/affinity_{id}.json'

    try:
        with open(file) as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0.0, 0.0
    boltzScore = data.get('affinity_pred_value')
    boltzProb = data.get('affinity_probability_binary')
    return boltzScore, boltzProb


def cal_boltzScore(affinity, likelihood):
    """
    Calculate Boltz score as max((-affinity-2)/4.0, 0.0)*likelihood.
    :param affinity: float, Boltz affinity score.
    :param likelihood: float, Boltz likelihood score.
    """
    return max((-affinity-2.0)/4.0, 0.0)*likelihood


def run(cmd):
    """
    Run a shell command and raise it if fails.
    :param cmd: str, shell command to run.
    """
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return completed.stdout


### Plot Boltz results
def plot_BoltzResult(input_file, boltzProb_column_name, boltzIC50_column_name, boltzScore_column_name,
                       trueIC50_column_name, cutoff, threshold, name, remove_inactive=False):
    """
    Plot experimental values versus Boltz-predicted results.
    :param input_file: str, path of the input file.
    :param boltzProb_column_name: str, name of the Boltz-predicted probability column.
    :param boltzIC50_column_name: str, name of the Boltz-predicted IC50 column.
    :param boltzScore_column_name: str, name of the Boltz score column.
    :param trueIC50_column_name: str, name of the true IC50 column.
    :param cutoff: float, IC50 cutoff.
    :param threshold: float, threshold for the Boltz-predicted probability.
    :param name: str, name of the plot.
    :param remove_inactive: bool, whether to remove inactive IC50 values.
    """
    df = pd.read_csv(input_file)
    folder = os.path.split(os.path.abspath(input_file))[0]

    # probability
    pred_prob = df[boltzProb_column_name].tolist()
    true_labels = df[trueIC50_column_name].apply(lambda x: int(x < cutoff)).tolist()
    output_file_PR = os.path.join(folder, f'{name}_PrecisionRecall.pdf')
    plot_PR(pred_prob, true_labels, num_points=100, title=f'Precision&Recall vs. Threshold for {name}', output_file=output_file_PR)
    pred_labels = [int(p >= threshold) for p in pred_prob]
    precision_1, recall_1, precision_0, recall_0 = calculate_PR(pred_labels, true_labels)
    print(f'Precision: {precision_1:.4f}, Recall: {recall_1:.4f} at threshold={threshold}')

    # IC50
    pred_IC50 = df[boltzIC50_column_name].tolist()
    true_IC50 = df[trueIC50_column_name].tolist()
    pred_boltzScore = df[boltzScore_column_name].tolist() if boltzScore_column_name is not None else []
    if remove_inactive:
        pred_IC50_removed, pred_boltzScore_removed, true_IC50_removed = [], [], []
        for i, t in enumerate(true_IC50):
            if t < cutoff:
                pred_IC50_removed.append(pred_IC50[i])
                true_IC50_removed.append(t)
                if boltzScore_column_name is not None:
                    pred_boltzScore_removed.append(pred_boltzScore[i])
        pred_IC50, pred_boltzScore, true_IC50 = pred_IC50_removed.copy(), pred_boltzScore_removed.copy(), true_IC50_removed.copy()
    output_file_IC50 = os.path.join(folder, f'{name}_IC50.pdf')
    print('Plot experimental IC50 value versus Boltz-predicted IC50:')
    plot_IC50(pred_IC50, true_IC50, title=f'Measured IC50 vs. Predicted IC50 for {name}', output_file=output_file_IC50)

    # Boltz score
    if boltzScore_column_name is not None:
        true_freeEnergy = ic50_to_freeEnergy(true_IC50)
        output_file_boltzScore = os.path.join(folder, f'{name}_boltzScore.pdf')
        print('Plot experimental binding free energy versus Boltz score:')
        plot_boltzScore(pred_boltzScore, true_freeEnergy, title=f'Binding free energy vs. Boltz score for {name}', output_file=output_file_boltzScore)


def calculate_PR(predicted_labels, true_labels):
    """
    Calculate precision and recall for single experiment.
    :param predicted_labels: list of ints, list of predicted labels, i.e., 1 or 0.
    :param true_labels: list of ints, list of true labels, i.e., 1, 0, Nan.
    :return: list of floats, precision_1, recall_1, precision_0, recall_0.
    """
    assert len(predicted_labels) == len(true_labels), 'Error: Number of predicted labels should be the same as that of true labels'
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, pred_label in enumerate(predicted_labels):
        true_label = true_labels[i]
        try:
            pred_label, true_label = int(pred_label), int(true_label)
            if pred_label == 1 and true_label == 1:
                TP += 1
            elif pred_label == 1 and true_label == 0:
                FP += 1
            elif pred_label == 0 and true_label == 0:
                TN += 1
            elif pred_label == 0 and true_label == 1:
                FN += 1
        except:
            continue

    # precision for label 1
    precision_1 = np.nan if TP + FP == 0 else TP * 1.0 / (TP + FP)
    # recall for label 1
    recall_1 = np.nan if TP + FN == 0 else TP * 1.0 / (TP + FN)
    # precision for label 0
    precision_0 = np.nan if TN + FN == 0 else TN * 1.0 / (TN + FN)
    # recall for label 0
    recall_0 = np.nan if TN + FP == 0 else TN * 1.0 / (TN + FP)

    return precision_1, recall_1, precision_0, recall_0


def ic50_to_freeEnergy(ic50, T=298.15):
    """
    Convert IC50 in uM to binding free energy ΔG in kcal/mol.
    :param ic50: list of floats, IC50 value in uM.
    :param T: float, temperature.
    """
    R_KCAL = 1.98720425864083e-3
    ic50 = np.asarray(ic50, dtype=float)
    if np.any(ic50 <= 0):
        raise ValueError("IC50 must be positive.")

    kd_M = ic50 * 1e-6  # assume IC50 ≈ Kd (binding assay / suitable conditions)
    return (R_KCAL * T) * np.log(kd_M)


def plot_PR(predictions, true_labels, num_points, title, output_file):
    """
    Plot precision recall curves for single experiment.
    :param predictions: list of floats, list of predicted values.
    :param true_labels: list of ints, list of true labels.
    :param num_points: int, number of points on the plot.
    :param title: str, title of the plot.
    :param output_file: str, path of the output file.
    """
    # calculate thresholds
    step = 1.0 / num_points
    thresholds = [step * i for i in range(num_points + 1)]

    # calculate precisions and recalls
    PR_dict = {'Precision_1': [], 'Recall_1': []}

    for threshold in thresholds:
        predicted_labels = [int(p >= threshold) for p in predictions]
        precision_1, recall_1, precision_0, recall_0 = calculate_PR(predicted_labels, true_labels)

        PR_dict['Precision_1'].append(precision_1)
        PR_dict['Recall_1'].append(recall_1)

    # plot
    plt.figure(1)
    plt.plot(thresholds, PR_dict['Precision_1'], label='Precision', color='blue')
    plt.plot(thresholds, PR_dict['Recall_1'], label='Recall', color='red')

    plt.grid(True)
    plt.xlabel('Thresholds', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)

    plt.legend(frameon=False, fontsize=12)
    plt.title(title, fontproperties=font)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_IC50(pred_IC50, true_IC50, title, output_file):
    """
    Plot experimental IC50 vs. predicted IC50.
    :param pred_IC50: list of floats, list of predicted IC50.
    :param true_IC50: list of floats, list of true IC50.
    :param title: str, title of the plot.
    :param output_file: str, path of the output file.
    """
    pred_IC50, true_IC50 = np.array(pred_IC50), np.array(true_IC50)

    lr = LinearRegression().fit(pred_IC50.reshape(-1, 1), true_IC50)
    slope, intercept = lr.coef_[0], lr.intercept_
    r2_fit = round(r2_score(true_IC50, lr.predict(pred_IC50.reshape(-1, 1))), 4)
    r_pearson = round(pearsonr(pred_IC50, true_IC50)[0], 4)
    print(f'R2 from linear regression: {r2_fit}, Pearson correlation coefficient: {r_pearson}')

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_IC50, true_IC50, c='r', marker='o')
    plt.xlabel('Predicted IC50 (uM)', fontproperties=font)
    plt.ylabel('Measured IC50 (uM)', fontproperties=font)
    x_line = np.linspace(pred_IC50.min(), pred_IC50.max(), 100)
    plt.plot(x_line, intercept + slope * x_line, c='b')
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.title(title, fontproperties=font)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_boltzScore(pred_boltzScore, true_freeEnergy, title, output_file):
    """
    Plot binding free energy vs. Boltz score.
    :param pred_boltzScore: list of floats, list of Boltz score.
    :param true_freeEnergy: list of floats, list of true binding free energy.
    :param title: str, title of the plot.
    :param output_file: str, path of the output file.
    """
    pred_boltzScore, true_freeEnergy = np.array(pred_boltzScore), np.array(true_freeEnergy)

    lr = LinearRegression().fit(pred_boltzScore.reshape(-1, 1), true_freeEnergy)
    slope, intercept = lr.coef_[0], lr.intercept_
    r2_fit = round(r2_score(true_freeEnergy, lr.predict(pred_boltzScore.reshape(-1, 1))), 4)
    r_pearson = round(pearsonr(pred_boltzScore, true_freeEnergy)[0], 4)
    print(f'R2 from linear regression: {r2_fit}, Pearson correlation coefficient: {r_pearson}')

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_boltzScore, true_freeEnergy, c='r', marker='o')
    plt.xlabel('Boltz score', fontproperties=font)
    plt.ylabel('Binding free energy (kcal/mol)', fontproperties=font)
    x_line = np.linspace(pred_boltzScore.min(), pred_boltzScore.max(), 100)
    plt.plot(x_line, intercept + slope * x_line, c='b')
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.title(title, fontproperties=font)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    ### Plot property distribution ###
    input_file = '../tests/test_plot_util.csv'
    plot_property_distribution(input_file, property_list=['TPSA'], method='hist')

    ### Plot docking score vs. MW ###
    # input_file = '../tests/test_plot_util.csv'
    # plot_dockingScore_vs_MW(input_file, smiles_column_name='Cleaned_SMILES', dockingScore_column_name='Docking_Score')
