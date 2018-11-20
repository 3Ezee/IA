#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:52:52 2018

@author: ezequiel
"""

from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs','action','reward','next_obs','done'])

class ExperienceMemory(object):
    """
    Un buffer que simula la memoria, experiencia del agente
    """
    def __init__(self, capacity = int(1e6)):
        """
        :param capacity: Capacidad total de la memoria cíclica (número máximo de experiencias almacenables)
        :return:
        """
        self.capacity = capacity
        self.memory_idx = 0#identificador que sabe la experiencia actual
        self.memory = []
        
    def sample(self, batch_size):
        """
        :param batch_size: tamaño de la memoria a recuperar
        :return: una muesta aleatoria del tamaño batch_size de experiencias aleatorias de la memoria
        """
        assert batch_size <= self.get_size(), "el tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        """
        :return: número de experiencias alamacenadas en memoria
        """
        return len(self.memory)
    
    def store(self, exp):
        """
        :param experience: objeto experiencia a ser almacenado en memoria
        :return:
        """
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1
        
        