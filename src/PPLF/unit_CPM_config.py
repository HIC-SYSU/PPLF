#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: unit_CPM_config.py
# Description:

import ml_collections

def get_config(image_size, feature_size, grid):
    config = ml_collections.ConfigDict()
    config.image_size = image_size
    config.feature_size = feature_size
    config.grid = grid
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    return config
