#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:15:00 2024

@author: guohan
"""

import os, sys
import pandas as pd
import numpy as np

from utils.tools import remove_unnamed_columns


### Preprocess docking score file ###
def preprocess_dockingScore(input_file_dockingScore, dockingScore_cutoff=0.0, **kwargs):
    """
    Preprocess docking score file
    :param input_file_dockingScore: str, path of the input docking score file
    :param dockingScore_cutoff: float, docking score cutoff
    :param id_column_name: str, name of the ID column
    :param dockingScore_column_name: str, name of the docking score column
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_dockingScore))[0] + '_DockingScore'

    # kwargs
    id_column_name = kwargs.get('id_column_name', 'ID')
    dockingScore_column_name = kwargs.get('dockingScore_column_name', 'r_i_docking_score')

    # read files
    df_dockingScore = pd.read_csv(input_file_dockingScore)
    df_dockingScore.rename(columns={id_column_name:'ID', dockingScore_column_name:'Docking_Score'}, inplace=True)
    df_dockingScore = pd.DataFrame(df_dockingScore, columns=['ID', 'Docking_Score'])
    print('Number of rows in the original docking score file:', df_dockingScore.shape[0])

    # round
    df_dockingScore['Docking_Score'] = df_dockingScore['Docking_Score'].apply(lambda score: np.round(score, decimals=3))
    # sort and deduplicate, keep the compounds with the lowest docking scores
    df_dockingScore.sort_values(by=['Docking_Score'], ascending=True, inplace=True)
    df_dockingScore = df_dockingScore.drop_duplicates(['ID'], keep='first', ignore_index=True)
    # filter
    df_dockingScore_filtered = df_dockingScore[df_dockingScore['Docking_Score'] <= dockingScore_cutoff]

    # write output file
    df_dockingScore_filtered = df_dockingScore_filtered.reset_index(drop=True)
    print('Number of rows in the filtered docking score file:', df_dockingScore_filtered.shape[0])
    df_dockingScore_filtered = remove_unnamed_columns(df_dockingScore_filtered)
    df_dockingScore_filtered.to_csv(f'{output_file}_{df_dockingScore_filtered.shape[0]}.csv')
    print('Preprocessing docking score file is done.')


### Combine SMILES input file and property input file ###
def add_property(input_file_SMILES, input_file_property, new_property_column_names, property_filters=None, **kwargs):
    """
    Filter compounds based on property cutoff
    :param input_file_SMILES: str, path of the input SMILES file
    :param input_file_property: str, path of the input property file
    :param new_property_column_names: list of strs, names of the property columns to be added in input_file_property
    :param property_filters: dict or None, dict of functions for property filters
    :param id_column_name_SMILESFile: str, name of the ID column in input_file_SMILES
    :param id_column_name_propertyFile: str, name of the ID column in input_file_property
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_property))[0] + '_Property'

    # kwargs
    id_column_name_SMILESFile = kwargs.get('id_column_name_SMILESFile', 'ID')
    id_column_name_propertyFile = kwargs.get('id_column_name_propertyFile', 'ID')

    if property_filters is None:
        property_filters = {}

    # read files
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES.rename(columns={id_column_name_SMILESFile: 'ID'}, inplace=True)
    print('Number of rows in SMILES file:', df_SMILES.shape[0])
    df_property = pd.read_csv(input_file_property)
    df_property.rename(columns={id_column_name_propertyFile:'ID'}, inplace=True)
    df_property = pd.DataFrame(df_property, columns=['ID']+new_property_column_names)
    print('Number of rows in property file:', df_property.shape[0])

    # merge
    df_merged = pd.merge(df_SMILES, df_property, how='left', on=['ID'])

    # filters
    for column, filter in property_filters.items():
        try:
            df_merged = df_merged[df_merged[column].apply(filter)]
        except Exception:
            print(f'Error: Filter for {column} column is not applied.')
            continue
    df_filtered = pd.DataFrame(df_merged, columns=df_SMILES.columns.tolist() + new_property_column_names)

    # write output file
    df_filtered = df_filtered.reset_index(drop=True)
    print('Number of rows in filtered property file:', df_filtered.shape[0])
    df_filtered = remove_unnamed_columns(df_filtered)
    df_filtered.to_csv(f'{output_file}_{df_filtered.shape[0]}.csv')
    print('Adding property is done.')


### Apply property filters ###
def filter_by_property(input_file, property_filters=None):
    """
    Filter compounds based on property cutoff
    :param input_file: str, path of the input file
    :param property_filters: dict or None, dict of functions for property filters
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file))[0] + '_Filtered'
    df = pd.read_csv(input_file)
    print('Number of rows in the original file:', df.shape[0])

    # filters
    if property_filters is None:
        property_filters = {}

    for column, filter in property_filters.items():
        try:
            df = df[df[column].apply(filter)]
        except Exception:
            print(f'Error: Filter in the {column} column is not applied.')
            continue

    # write output file
    df = df.reset_index(drop=True)
    print('Number of rows in the filtered file:', df.shape[0])
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')
    print('Applying filters is done.')



if __name__ == '__main__':

    ### Preprocess docking score file ###
    # input_file_dockingScore = 'tests/test_preprocess_dockingScore.csv'
    # dockingScore_cutoff = 0.0
    # preprocess_dockingScore(input_file_dockingScore, dockingScore_cutoff, id_column_name='ID', dockingScore_column_name ='r_i_docking_score')

    ### Combine SMILES input file and property input file ###
    input_file_SMILES = 'tests/test_SMILES_file.csv'
    input_file_property = 'tests/test_property_file.csv'
    new_property_column_names = ['MW', 'logP', 'HBD']
    property_filters = {'MW':lambda x: x <= 650, 'logP':lambda x: x <= 5.0}
    add_property(input_file_SMILES, input_file_property, new_property_column_names, property_filters,
                 id_column_name_SMILESFile = 'ID', id_column_name_propertyFile = 'ID')

    ### Apply property filters ###
    input_file = 'tests/test_property_file_Property_295.csv'
    property_filters = {'MW': lambda x: x <= 450, 'logP': lambda x: x <= 4.5}
    filter_by_property(input_file, property_filters)