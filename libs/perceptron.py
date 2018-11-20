#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:42:49 2018

@author: ezequiel
"""

import torch

class SLP(torch.nn.Module):
    """
    Single Layer Perceptron / Neurona de una sola capa para aproximar funciones
    """
    
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        :param input_shape: tamaño o forma de los datos de entrada
        :param output_shape: tamaño o forma de los datos de salida
        :param device: dispositivo ('cpu' o 'cuda') que la spl debe utilizar para almacenar los inputs en cada iteración
        """
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape) 
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)
        
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x)) ##función de activación RELU
        x = self.out(x)
        return x
    