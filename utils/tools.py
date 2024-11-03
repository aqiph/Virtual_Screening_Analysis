#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:55:00 2023

@author: guohan

"""

import os
import pandas as pd
import numpy as np
import base64


def remove_unnamed_columns(df):
    """
    remove unnamed columns
    """
    unnamed_cols = df.columns.str.contains('Unnamed:')
    unnamed_cols_name = df.columns[unnamed_cols]
    df.drop(unnamed_cols_name, axis=1, inplace=True)
    return df


def png_to_base64(image_path):
    """
    convert a figure in .png to base64
    :param image_path: str, path of the input image.
    :return: str, base64 string of the given image.
    """
    try:
        with open(image_path, 'rb') as image_file:
            image_str = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(e)
        image_str = ''

    if len(image_str) == 0:
        return ''
    else:
        return f'<img src="data:image/png;base64,{image_str}">'

