#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:15:00 2024

@author: guohan
"""

import os, sys
import pandas as pd
import numpy as np

from tools import remove_unnamed_columns


### Apply docking score filter ###
def filter_by_dockingScore(input_file_SMILES, input_file_dockingScore, dockingScore_cutoff=0.0, **kwargs):
    """
    Filter compounds based on docking score
    :param input_file_SMILES: str, path of the input SMILES file
    :param input_file_dockingScore: str, path of the input docking score file
    :param dockingScore_cutoff: float, docking score cutoff
    :param id_column_name_SMILESFile: str, name of the ID column in input_file_SMILES
    :param id_column_name_dockingScoreFile: str, name of the ID column in input_file_dockingScore
    :param dockingScore_column_name_dockingScoreFile: str, name of the docking score column in input_file_dockingScore
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_SMILES))[0] + '_DockingScore'

    # kwargs
    id_column_name_SMILESFile = kwargs.get('id_column_name_SMILESFile', 'ID')
    id_column_name_dockingScoreFile = kwargs.get('id_column_name_dockingScoreFile', 'ID')
    dockingScore_column_name_dockingScoreFile = kwargs.get('dockingScore_column_name_dockingScoreFile', 'r_i_docking_score')

    # read files
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES.rename(columns={id_column_name_SMILESFile:'ID'}, inplace=True)
    print('Number of rows in SMILES file:', df_SMILES.shape[0])
    df_dockingScore = pd.read_csv(input_file_dockingScore)
    df_dockingScore.rename(columns={id_column_name_dockingScoreFile:'ID', dockingScore_column_name_dockingScoreFile:'Docking_Score'}, inplace=True)
    df_dockingScore = pd.DataFrame(df_dockingScore, columns=['ID', 'Docking_Score'])
    print('Number of rows in docking score file:', df_dockingScore.shape[0])

    # round
    df_dockingScore['Docking_Score'] = df_dockingScore['Docking_Score'].apply(lambda score: np.round(score, decimals=3))
    # sort and deduplicate, keep the compounds with the lowest docking scores
    df_dockingScore.sort_values(by=['Docking_Score'], ascending=True, inplace=True)
    df_dockingScore = df_dockingScore.drop_duplicates(['ID'], keep='first', ignore_index=True)
    # filter
    df_dockingScore_filtered = df_dockingScore[df_dockingScore['Docking_Score'] <= dockingScore_cutoff]
    # merge
    df_filtered = pd.merge(df_dockingScore_filtered, df_SMILES, how='inner', on=['ID'])
    df_filtered = pd.DataFrame(df_filtered, columns=df_SMILES.columns.tolist()+['Docking_Score'])

    # write output file
    df_filtered = df_filtered.reset_index(drop=True)
    print('Number of rows in the filtered file:', df_filtered.shape[0])
    df_filtered = remove_unnamed_columns(df_filtered)
    df_filtered.to_csv(f'{output_file}_{df_filtered.shape[0]}.csv')
    print('Applying docking score filter is done.')


### Apply property filters ###
def filter_by_property(input_file_SMILES, input_file_property, property_filters=None, **kwargs):
    """
    Filter compounds based on property cutoff
    :param input_file_SMILES: str, path of the input SMILES file
    :param input_file_property: str, path of the input property file
    :param property_filters: dict or None, dict of functions for property filters
    :param id_column_name_SMILESFile: str, name of the ID column in input_file_SMILES
    :param id_column_name_propertyFile: str, name of the ID column in input_file_property
    :param property_column_names_propertyFile: list of strs, names of the property columns in input_file_property
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_SMILES))[0] + '_Property'

    # kwargs
    id_column_name_SMILESFile = kwargs.get('id_column_name_SMILESFile', 'ID')
    id_column_name_propertyFile = kwargs.get('id_column_name_propertyFile', 'ID')
    property_column_names_propertyFile = kwargs.get('property_column_names_propertyFile', [])

    if property_filters is None:
        property_filters = {}

    # read files
    df_SMILES = pd.read_csv(input_file_SMILES)
    df_SMILES.rename(columns={id_column_name_SMILESFile: 'ID'}, inplace=True)
    print('Number of rows in SMILES file:', df_SMILES.shape[0])
    df_property = pd.read_csv(input_file_property)
    df_property.rename(columns={id_column_name_propertyFile:'ID'}, inplace=True)
    df_property = pd.DataFrame(df_property, columns=['ID']+property_column_names_propertyFile)
    print('Number of rows in property file:', df_property.shape[0])

    # filters
    for column, filter in property_filters.items():
        try:
            df_property = df_property[df_property[column].apply(filter)]
        except Exception:
            print(f'Error: Filter for {column} column is not applied.')
            continue
    # merge
    df_filtered = pd.merge(df_property, df_SMILES, how='inner', on=['ID'])
    df_filtered = pd.DataFrame(df_filtered, columns=df_SMILES.columns.tolist()+property_column_names_propertyFile)

    # write output file
    df_filtered = df_filtered.reset_index(drop=True)
    print('Number of rows in filtered property file:', df_filtered.shape[0])
    df_filtered = remove_unnamed_columns(df_filtered)
    df_filtered.to_csv(f'{output_file}_{df_filtered.shape[0]}.csv')
    print('Applying property filters is done.')




if __name__ == '__main__':

    ### Apply docking score filter ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_dockingScore = 'tests/test_dockingScore_filter.csv'
    # dockingScore_cutoff = 0.0
    # filter_by_dockingScore(input_file_SMILES, input_file_dockingScore, dockingScore_cutoff,
    #                        id_column_name_SMILESFile='ID', id_column_name_dockingScoreFile='ID',
    #                        dockingScore_column_name_dockingScoreFile='r_i_docking_score')


    ### Apply property filters ###
    input_file_SMILES = 'tests/test_SMILES_file.csv'
    input_file_property = 'tests/test_property_file.csv'
    property_filters = {'MW':lambda x: x <= 650, 'logP':lambda x: x <= 5.0}
    filter_by_property(input_file_SMILES, input_file_property, property_filters,
                       id_column_name_SMILESFile = 'ID', id_column_name_propertyFile = 'ID',
                       property_column_names_propertyFile = ['MW', 'logP', 'HBD'])