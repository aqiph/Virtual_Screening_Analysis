#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:15:00 2024

@author: guohan
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import gemmi

from utils.util import get_boltzResult, cal_boltzScore, run, plot_BoltzResult
from utils.tools import remove_unnamed_columns


### Preprocess docking score file ###
def preprocess_dockingScore(input_file_dockingScore, dockingScore_cutoff=0.0, **kwargs):
    """
    Preprocess docking score file.
    :param input_file_dockingScore: str, path of the input docking score file.
    :param dockingScore_cutoff: float, docking score cutoff.
    :param id_column_name: str, name of the ID column.
    :param dockingScore_column_name: str, name of the docking score column.
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


### Preprocess boltz2 binding affinity files ###
def preprocess_boltzResult(input_file_SMILES, input_dir_boltzResult, **kwargs):
    """
    Preprocess Boltz results.
    :param input_file_SMILES: str, path of the input SMILES file.
    :param input_dir_boltzResult: str, path of the input Boltz result file.
    :param id_column_name: str, name of the ID column.
    :param boltzResult_column_name: str, name of the Boltz result column.
    :param output_type: str, output type for binding affinity, allowed values include: 'IC50', 'binding_free_energy'.
    :param calculate_boltz_score: bool, whether calculate Boltz score or not.
    :param process_binding_pose: bool, whether process binding pose or not.
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file_SMILES))[0] + '_BoltzResult'

    # kwargs
    id_column_name = kwargs.get('id_column_name', 'ID')
    boltzResult_column_name = kwargs.get('boltzResult_column_name', 'Boltz')
    boltzProb_column_name = 'Probability_' + boltzResult_column_name
    output_type = kwargs.get('output_type', 'IC50')   # 'IC50' or 'binding_free_energy'
    boltzScore = kwargs.get('calculate_boltz_score', False)
    process_pose = kwargs.get('process_binding_pose', False)

    # extract Boltz2 binding affinity
    df = pd.read_csv(input_file_SMILES)
    df[boltzResult_column_name] = df[id_column_name].apply(lambda id: get_boltzResult(id, input_dir_boltzResult))
    df[[boltzResult_column_name, boltzProb_column_name]] = pd.DataFrame(df[boltzResult_column_name].values.tolist())

    # compute Boltz score
    if boltzScore:
        boltzScore_column_name = 'BoltzScore_' + boltzResult_column_name
        df[boltzScore_column_name] = df.apply(lambda row: cal_boltzScore(row[boltzResult_column_name], row[boltzProb_column_name]), axis=1)

    # change binding affinity type
    if output_type == 'IC50':   # units: uM
        df[boltzResult_column_name] = np.power(10, df[boltzResult_column_name])
        boltzAffinity_column_name = 'IC50_' + boltzResult_column_name + ' (uM)'
    elif output_type.lower() == 'binding_free_energy':
        df[boltzResult_column_name] = -(6.0 - df[boltzResult_column_name]) * 1.364
        boltzAffinity_column_name = 'Binding_Free_Energy' + boltzResult_column_name + ' (kcal/mol)'
    else:
        raise ValueError('output_type')
    df.rename(columns={boltzResult_column_name:boltzAffinity_column_name}, inplace=True)
    df[boltzAffinity_column_name] = df[boltzAffinity_column_name].apply(lambda score: np.round(score, decimals=3))
    df[boltzProb_column_name] = df[boltzProb_column_name].apply(lambda score: np.round(score, decimals=3))

    # write output file
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')

    # extract and combine binding modes
    if not process_pose:
        return
    IDs = df[id_column_name].tolist()
    mae_files = []
    for idx in tqdm(IDs, desc="Converting CIF to MAE"):
        cif_file = f'{input_dir_boltzResult}/predictions/{idx}/{idx}_model_0.cif'
        mae_file = f'{input_dir_boltzResult}/predictions/{idx}/{idx}_model_0.mae'
        doc = gemmi.cif.read(cif_file)
        block = doc.sole_block()
        block.name = idx
        block.set_pair('_entry.id', idx)
        doc.write_file(cif_file)
        cmd = ['bash', '/opt/schrodinger/suites2023-4/utilities/structconvert', cif_file, mae_file]
        run(cmd)
        mae_files.append(mae_file)

    cat_cmd = ['bash', '/opt/schrodinger/suites2023-4/utilities/structcat', '-o', output_file+'.mae']
    cat_cmd += mae_files
    run(cat_cmd)


### Combine SMILES input file and property input file ###
def add_property(input_file_SMILES, input_file_property, new_property_column_names, property_filters=None, **kwargs):
    """
    Filter compounds based on property cutoff.
    :param input_file_SMILES: str, path of the input SMILES file.
    :param input_file_property: str, path of the input property file.
    :param new_property_column_names: list of strs, names of the property columns to be added in input_file_property.
    :param property_filters: dict or None, dict of functions for property filters.
    :param id_column_name_SMILESFile: str, name of the ID column in input_file_SMILES.
    :param id_column_name_propertyFile: str, name of the ID column in input_file_property.
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
    Filter compounds based on property cutoff.
    :param input_file: str, path of the input file.
    :param property_filters: dict or None, dict of functions for property filters, i.e., {column: function}.
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


def filter_by_dockingScore(input_file, dockingScore_cutoffs=None):
    """
    Filter compounds based on docking score cutoff.
    :param input_file: str, path of the input file.
    :param dockingScore_cutoffs: dict or None, dict of docking score cutoffs for docking filters, i.e., {column: cutoff}.
    :return: None
    """
    # files
    output_file = os.path.splitext(os.path.abspath(input_file))[0] + '_Filtered'
    df = pd.read_csv(input_file)
    print('Number of rows in the original file:', df.shape[0])

    # filters
    if dockingScore_cutoffs is not None:
        df = df[df.apply(lambda row:select_by_dockingScore(row, dockingScore_cutoffs), axis=1)]

    # write output file
    df = df.reset_index(drop=True)
    print('Number of rows in the filtered file:', df.shape[0])
    df = remove_unnamed_columns(df)
    df.to_csv(f'{output_file}_{df.shape[0]}.csv')
    print('Applying docking score filters is done.')


def select_by_dockingScore(row, dockingScore_cutoffs):
    """
    Helper function for filter_by_dockingScore, return True if any docking score satisfies the requirement, return False otherwise.
    :param row: row of pd.DataFrame object.
    :param dockingScore_cutoffs: dict or None, dict of docking score cutoffs for docking filters, i.e., {column: cutoff}.
    :return: bool
    """
    if dockingScore_cutoffs is None:
        return True

    for column, cutoff in dockingScore_cutoffs.items():
        try:
            if row[column] <= cutoff:
                return True
        except Exception:
            print(f'Error:Docking score filter in the {column} column is not applied.')
    return False



if __name__ == '__main__':

    ### Preprocess docking score file ###
    # input_file_dockingScore = 'tests/test_preprocess_dockingScore.csv'
    # dockingScore_cutoff = 0.0
    # preprocess_dockingScore(input_file_dockingScore, dockingScore_cutoff, id_column_name='ID', dockingScore_column_name ='r_i_docking_score')

    ### Preprocess boltz2 binding affinity files ###
    input_file_SMILES = 'tests/test_boltz.csv'
    input_dir_boltzResult = 'tests/test_boltz_results'
    preprocess_boltzResult(input_file_SMILES, input_dir_boltzResult, id_column_name='ID', boltzResult_column_name='pfTopo I',
                          output_type='IC50', calculate_boltz_score=True, process_binding_pose=False)

    input_file = 'tests/test_boltz_BoltzResult_5.csv'
    boltzProb_column_name = 'Probability_pfTopo I'
    boltzIC50_column_name = 'IC50_pfTopo I (uM)'
    boltzScore_column_name = 'BoltzScore_pfTopo I'
    trueIC50_column_name = 'IC50(μM)_pfTopoI'
    plot_BoltzResult(input_file, boltzProb_column_name, boltzIC50_column_name, boltzScore_column_name,
                       trueIC50_column_name, cutoff=50, threshold=0.5, name='pfTopo I', remove_inactive=False)

    ### Combine SMILES input file and property input file ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_property = 'tests/test_property_file.csv'
    # new_property_column_names = ['MW', 'logP', 'HBD']
    # property_filters = {'MW':lambda x: x <= 650, 'logP':lambda x: x <= 5.0}
    # add_property(input_file_SMILES, input_file_property, new_property_column_names, property_filters,
    #              id_column_name_SMILESFile = 'ID', id_column_name_propertyFile = 'ID')

    ### Apply property filters ###
    # input_file = 'tests/test_property_file_Property_295.csv'
    # property_filters = {'MW': lambda x: x <= 450, 'logP': lambda x: x <= 4.5}
    # filter_by_property(input_file, property_filters)

    # input_file = 'tests/test_filter_by_dockingScore_998.csv'
    # dockingScore_cutoff = {'Docking_Score_Pocket1': -5.0, 'Docking_Score_Pocket3': -8.0}
    # filter_by_dockingScore(input_file, dockingScore_cutoff)
