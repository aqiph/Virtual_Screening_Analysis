#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:00 2024

@author: guohan
"""

import os, sys
import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
from chembl_structure_pipeline import *
from chembl_structure_pipeline.checker import *


from tools import remove_unnamed_columns


### Get cluster labels ###
def get_clusterLabel(input_file_SMILES, input_file_clusterLabel, **kwargs):
    """
    Get cluster labels
    :param input_file_SMILES: str, path of the input SMILES file
    :param input_file_clusterLabel: str, path of the input cluster label file
    :param id_column_name_SMILESFile: str, name of the ID column in input_file_SMILES
    :param id_column_name_clusterLabelFile: str, name of the ID column in input_file_clusterLabel
    :param clusterLabel_column_name_clusterLabelFile: str, name of the cluster label column in input_file_clusterLabel
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_SMILES))[0] + '_ClusterLabel'

    # kwargs
    id_column_name_SMILESFile = kwargs.get('id_column_name_SMILESFile', 'ID')
    id_column_name_clusterLabelFile = kwargs.get('id_column_name_clusterLabelFile', 'ID')
    clusterLabel_column_name_clusterLabelFile = kwargs.get('clusterLabel_column_name_clusterLabelFile', 'MCS Cluster')

    # read files
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES.rename(columns={id_column_name_SMILESFile:'ID'}, inplace=True)
    print('Number of rows in SMILES file:', df_SMILES.shape[0])
    df_clusterLabel = pd.read_csv(input_file_clusterLabel)
    df_clusterLabel.rename(columns={id_column_name_clusterLabelFile:'ID', clusterLabel_column_name_clusterLabelFile:'MCS_Cluster'}, inplace=True)
    df_clusterLabel = pd.DataFrame(df_clusterLabel, columns=['ID', 'MCS_Cluster'])
    print('Number of rows in cluster label file:', df_clusterLabel.shape[0])

    # merge
    df = pd.merge(df_SMILES, df_clusterLabel, how='inner', on=['ID'])
    df = pd.DataFrame(df, columns=df_SMILES.columns.tolist() + ['MCS_Cluster'])

    # write output file
    df = df.reset_index(drop=True)
    print('Number of rows in the output file:', df.shape[0])
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')
    print('Getting cluster label is done.')


### Select representatives ###
def select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method, **kwargs):
    """
    Select compounds from each cluster.
    :param input_file: str, path of the input file.
    :param clusterLabel_column_name: str, name of the cluster label column.
    :param selection_method: str, method to select representatives, allowed values include 'best', 'smallest', 'centroid_by_MCS'.

    :param outlier_label: label of outliers.
    :param keep_outlier: bool, whether to keep outliers or not.
    :param count_per_cluster: int, larger than 1, number of compounds from each cluster.
    :param percentage_per_cluster: float, less than 1 and larger than 0, percentage of compounds from each cluster.

    :param dockingScore_column_name: str, name of the docking score column.
    :param property_rules: dict, dictionary of property rules, each key-value is defined as column_name:function.
    :param min_num_rules: int, minimum number of rules a compound need to satisfy.
    :param SMILES_column_name: str, name of the SMILES column.
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file))[0]+'_representative'

    # read input file and define DataFrame for representatives
    df = pd.read_csv(input_file)
    df_representative = None

    # get cluster labels and outlier label
    clusterLabel_set = set(df[clusterLabel_column_name])
    outlier_label = kwargs.get('outlier_label', 0)
    keep_outlier = kwargs.get('or not', True)

    # process clusters
    for label in clusterLabel_set:
        print(f'Cluster label={label}...')
        df_cluster = df[df[clusterLabel_column_name] == label]
        # get request count for outliers
        if label == outlier_label:
            if keep_outlier:
                count = df_cluster.shape[0]   # keep outliers
            else:
                continue
        # get request count for other clusters
        else:
            if 'count_per_cluster' in kwargs:
                count = kwargs.get('count_per_cluster', 0)   # get the request number of compounds per cluster, default is 0
                print(f'Request count number is {count}.')
            elif 'percentage_per_cluster' in kwargs:
                percentage = kwargs.get('percentage_per_cluster', 0.0)
                count = round(df_cluster.shape[0] * percentage)
                if keep_outlier:
                    count = max(count, 1)
            else:
                raise Exception('Error: Invalid number of compounds per cluster, please define either count_per_cluster or percentage_per_cluster.')

        ## get representatives from each cluster
        df_new_representative = pd.DataFrame()
        # get the best compounds (meet the most criteria and have the highest docking score)
        if selection_method == 'best':
            dockingScore_column_name = kwargs.get('dockingScore_column_name', 'Docking_Score')
            property_rules = kwargs.get('property_rules', {})
            min_num_rules = kwargs.get('min_num_rules', len(property_rules))
            df_new_representative = get_best_compound(df_cluster, dockingScore_column_name, property_rules, min_num_rules, count)
        # get the smallest compounds (the least heavy atoms)
        elif selection_method == 'smallest':
            SMILES_column_name = kwargs.get('SMILES_column_name', 'SMILES')
            dockingScore_column_name = kwargs.get('dockingScore_column_name', 'Docking_Score')
            df_new_representative = get_smallest_compound(df_cluster, SMILES_column_name, dockingScore_column_name, count)
        elif selection_method == 'centroid_by_MCS':
            id_column_name = kwargs.get('id_column_name', 'ID')
            SMILES_column_name = kwargs.get('SMILES_column_name', 'SMILES')
            df_new_representative = get_centroid_by_mcs(df_cluster, id_column_name, SMILES_column_name, clusterLabel_column_name, label, label==outlier_label)

        # concat representative df from each cluster
        if df_new_representative.shape[0] == 0:
            continue
        if df_representative is None:
            df_representative = df_new_representative
        else:
            df_representative = pd.concat([df_representative, df_new_representative], ignore_index=True, sort=False)

    # write output file
    df_representative = df_representative.reset_index(drop=True)
    print('Number of rows in the output file:', df_representative.shape[0])
    df_representative = remove_unnamed_columns(df_representative)
    df_representative.to_csv(f'{output_file}_{df_representative.shape[0]}.csv')
    print('Selection of representative compounds is done.')


def get_best_compound(df_cluster, dockingScore_column_name, property_rules, min_num_rules, count):
    """
    Helper function for select_cmpds_from_clusters. Select the best compounds from each cluster,
    which meet the most criteria and have the highest docking score.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param dockingScore_column_name: str, name of the docking score column.
    :param property_rules: dict, dictionary of property rules, each key-value is defined as column_name:function.
    :param min_num_rules: int, minimum number of rules a compound need to satisfy.
    :param count: int, number of compounds from each cluster.
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    COLUMNS = df_cluster.columns.tolist()

    # convert value to score (1 or 0) for each property
    for column, rule in property_rules.items():
        try:
            df_cluster.insert(df_cluster.shape[1], column+'_', [int(v) for v in df_cluster[column].apply(rule)])
        except Exception:
            print(f'Error: Rule for {column} column is not applied.')
            continue

    # get property score column names
    property_column_names = property_rules.keys()
    property_score_column_names = [property+'_'for property in property_column_names]
    # compute compound property score (number of criteria met)
    df_cluster.insert(df_cluster.shape[1], 'Property_Score', df_cluster[property_score_column_names].sum(axis=1).values.tolist())
    # get compounds that meet minimum number of criteria
    df_cluster = df_cluster[df_cluster['Property_Score'] >= min_num_rules]
    # sort compounds based on property score first then on docking score
    df_cluster = df_cluster.sort_values(by=['Property_Score', dockingScore_column_name], ascending=[False, True], ignore_index=True)
    # get representive compounds based on count
    n = min(count, df_cluster.shape[0])
    df_subset = df_cluster.loc[0:(n-1)]
    df_subset = pd.DataFrame(df_subset, columns=COLUMNS+['Property_Score'])

    return df_subset


def get_smallest_compound(df_cluster, SMILES_column_name, dockingScore_column_name, count):
    """
    Helper function for select_cmpds_from_clusters. Select the smallest compounds.
    (i.e., the least heavy atoms) from each cluster.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param SMILES_column_name: str, name of the SMILES column.
    :param dockingScore_column_name: str, name of the docking score column.
    :param count: int, number of compounds from each cluster.
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    COLUMNS = df_cluster.columns.tolist()

    # compute the number of heavy atoms
    NHA = [Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in df_cluster[SMILES_column_name].values.tolist()]
    df_cluster.insert(len(COLUMNS), 'NHA', NHA)
    # sort compounds based on number of heavy atoms first then on docking score
    df_cluster = df_cluster.sort_values(by=['NHA', dockingScore_column_name], ascending=[True, True], ignore_index=True)
    # get representive compounds based on count
    n = min(count, df_cluster.shape[0])
    df_subset = df_cluster.loc[0:(n-1)]
    df_subset = pd.DataFrame(df_subset, columns=COLUMNS)

    return df_subset


def get_centroid_by_mcs(df_cluster, id_column_name, SMILES_column_name, clusterLabel_column_name, label, is_outlier):
    """
    Helper function for select_cmpds_from_clusters.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param id_column_name: str, name of the ID column
    :param SMILES_column_name: str, name of the SMILES column.
    :param clusterLabel_column_name: str, name of the cluster label column.
    :param label: label of the cluster.
    :param is_outlier: bool, whether this cluster is outliers.
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    # return outliers directly
    if is_outlier:
        return df_cluster

    # compute MCS
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in df_cluster[SMILES_column_name].values.tolist()]
    for mol in mol_list:
        Chem.Kekulize(mol)   # Use Kekule form to distinguish between N and [nH] in aromatic rings.

    mcs_SMARTS = rdFMCS.FindMCS(mol_list, ringMatchesRingOnly=True, completeRingsOnly=True,
                                bondCompare=rdFMCS.BondCompare.CompareOrderExact).smartsString

    # convert MCS from SMARTS format to SMILES format
    mcs_mol = Chem.MolFromSmarts(mcs_SMARTS)    # Not sanitize mol when coverting SMARTS to mol
    Chem.SanitizeMol(mcs_mol)
    mcs_SMILES = Chem.MolToSmiles(mcs_mol)
    try:
        mcs_SMILES = Chem.MolToSmiles(Chem.MolFromSmiles(mcs_SMILES, sanitize=True))
    except Exception:
        print('Error: Cannot convert pattern to proper SMILES, using pattern instead.')
    print(mcs_SMILES)

    # centroid DataFrame
    df_centroid = pd.DataFrame({id_column_name:[f'cluster{label}_MCS_centroid'], SMILES_column_name:[mcs_SMILES],
                                clusterLabel_column_name:[label]})

    return df_centroid


def get_centroid_by_mcs_obsolete(df_cluster, SMILES_column_name, dockingScore_column_name, count, is_outlier):
    """
    Helper function for select_cmpds_from_clusters.
    :param df_cluster: pd.DataFrame object which contains compounds from each cluster.
    :param SMILES_column_name: str, name of the SMILES column.
    :param count: int, number of compounds from each cluster.
    :param is_outlier: bool, whether this cluster is outliers.
    :return: pd.DataFrame object which contains selected representative compounds.
    """
    # return outliers directly
    if is_outlier:
        return df_cluster

    COLUMNS = df_cluster.columns.tolist()

    # compute the list of MCS between all pairs
    MCS_list = []
    SMILES_list = df_cluster[SMILES_column_name].values.tolist()
    SMILES_num = len(SMILES_list)
    for i in range(SMILES_num):
        try:
            mol_1 = Chem.MolFromSmiles(SMILES_list[i])
        except Exception:
            print(f"Error: Invalid SMILES {SMILES_list[i]}.")
            continue
        for j in range(i+1, SMILES_num):
            try:
                mol_2 = Chem.MolFromSmiles(SMILES_list[j])
            except Exception:
                print(f"Error: Invalid SMILES {SMILES_list[j]}.")
                continue
            mcs = rdFMCS.FindMCS([mol_1, mol_2], completeRingsOnly=True, timeout=1).smartsString
            mcs_SMILES = Chem.MolToSmiles(Chem.MolFromSmarts(mcs))
            MCS_list.append(mcs_SMILES)
    # compute the most common MCS
    MCS_counter = Counter(MCS_list)
    most_common_MCS = MCS_counter.most_common(1)[0][0]
    most_common_MCS = Chem.MolToSmiles(Chem.MolFromSmiles(most_common_MCS))
    print(most_common_MCS)
    return df_cluster




if __name__ == '__main__':
    ### Get cluster labels ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_clusterLabel = 'tests/test_clusterLabel_file.csv'
    # get_clusterLabel(input_file_SMILES, input_file_clusterLabel,
    #                  id_column_name_SMILESFile='ID', id_column_name_clusterLabelFile='ID',
    #                  clusterLabel_column_name_clusterLabelFile='MCS Cluster 0.7')


    ### Select representatives ###
    # print(pd.options.mode.copy_on_write)
    input_file = 'tests/test_select_representatives.csv'
    clusterLabel_column_name = 'MCS_Cluster'
    # Select the best compounds #
    # property_rules = {'MW':lambda mw: mw <= 500, 'logP':lambda logp: logp > 0.0, 'HBD': lambda hbd: hbd <= 3}
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='best',
    #                            outlier_label=0, keep_outlier=True,
    #                            count_per_cluster=3, dockingScore_column_name='Docking_Score',
    #                            property_rules=property_rules, min_num_rules=3)
    # Select the smallest compound #
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='smallest',
    #                            outlier_label=0, keep_outlier=True,
    #                            count_per_cluster=2, dockingScore_column_name='Docking_Score',
    #                            SMILES_column_name='Cleaned_SMILES')
    # Get the centroid defined as MCS #
    select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='centroid_by_MCS',
                               outlier_label=0, keep_outlier=True,
                               count_per_cluster=1, SMILES_column_name='Cleaned_SMILES')


