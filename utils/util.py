#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 08 14:32:00 2025

@author: guohan

"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


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


if __name__ == '__main__':
    ### Plot property distribution ###
    input_file = '../tests/test_plot_util.csv'
    plot_property_distribution(input_file, property_list=['TPSA'], method='hist')

    ### Plot docking score vs. MW ###
    # input_file = '../tests/test_plot_util.csv'
    # plot_dockingScore_vs_MW(input_file, smiles_column_name='Cleaned_SMILES', dockingScore_column_name='Docking_Score')
