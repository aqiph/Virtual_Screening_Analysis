#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:19:00 2024

@author: guohan
"""

import os, sys
path_list = sys.path
module_path = '/Users/guohan/Documents/Codes/Virtual_Screening_Analysis'
if module_path not in sys.path:
    sys.path.append(module_path)
    print('Add module path')

import pandas as pd
import numpy as np

from compound_filtering import preprocess_dockingScore, add_property, filter_by_property, filter_by_dockingScore, preprocess_boltzResult
from compound_selection import get_clusterLabel, select_cmpds_from_clusters, add_centroid_figure_column
from utils.util import plot_boltzResult


if __name__=='__main__':

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
    plot_boltzResult(input_file, boltzProb_column_name, boltzIC50_column_name, boltzScore_column_name,
                     trueIC50_column_name, cutoff=50, threshold=0.5, name='pfTopo I', remove_inactive=False)

    ### Combine SMILES input file and property input file ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_property = 'tests/test_property_file.csv'
    # new_property_column_names = ['MW', 'logP', 'HBD']
    # property_filters = {'MW': lambda x: x <= 650, 'logP': lambda x: x <= 5.0}
    # add_property(input_file_SMILES, input_file_property, new_property_column_names, property_filters,
    #              id_column_name_SMILESFile='ID', id_column_name_propertyFile='ID')

    ### Apply property filters ###
    # input_file = 'tests/test_property_file_Property_295.csv'
    # property_filters = {'MW': lambda x: x <= 650, 'logP': lambda x: x <= 4.5}
    # filter_by_property(input_file, property_filters)

    # input_file = 'tests/test_filter_by_dockingScore_998.csv'
    # dockingScore_cutoff = {'Docking_Score_Pocket1': -5.0, 'Docking_Score_Pocket3': -8.0}
    # filter_by_dockingScore(input_file, dockingScore_cutoff)

    ### Get cluster labels ###
    # input_file_SMILES = 'tests/test_SMILES_file.csv'
    # input_file_clusterLabel = 'tests/test_clusterLabel_file.csv'
    # get_clusterLabel(input_file_SMILES, input_file_clusterLabel,
    #                  id_column_name_SMILESFile='ID', id_column_name_clusterLabelFile='ID',
    #                  clusterLabel_column_name_clusterLabelFile='MCS Cluster 0.7')

    ### Select representatives ###
    input_file = 'tests/test_select_representatives.csv'
    clusterLabel_column_name = 'MCS_Cluster'
    # Select the best compounds #
    # property_rules = {'MW': lambda mw: mw <= 500, 'logP': lambda logp: logp > 0.0, 'HBD': lambda hbd: hbd <= 3}
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='best',
    #                            outlier_label=0, keep_outlier=True, count_per_cluster=2,
    #                            SMILES_column_name='Cleaned_SMILES', dockingScore_column_name='Docking_Score',
    #                            property_rules=property_rules, min_num_rules=2, addCentroid=True)
    # Select the smallest compound #
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='smallest',
    #                            outlier_label=0, keep_outlier=True, count_per_cluster=1,
    #                            SMILES_column_name='Cleaned_SMILES', dockingScore_column_name='Docking_Score',
    #                            addCentroid=True)
    # Get the centroid defined as MCS #
    # select_cmpds_from_clusters(input_file, clusterLabel_column_name, selection_method='centroid_by_MCS',
    #                            outlier_label=0, keep_outlier=True,
    #                            id_column_name = 'ID', SMILES_column_name='Cleaned_SMILES')

    ### Add centroid figure column ###
    # input_file = 'tests/test_add_centroid_figure_column.csv'
    # add_centroid_figure_column(input_file, id_column_name='ID', SMILES_column_name='Cleaned_SMILES',
    #                            centroid_column_name='MCS_Centroid')
