#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:13:22 2020

@author: alisonpitceathly
"""
##############################################################################
#                   Coding Project
##############################################################################

##############################################################################
#                Goal 1
##############################################################################

class Deuteron_Data:
    """
    This Class represents the Particle data for protons, neutrons, anti-protons
    and anti-neutrons. 
    """
    def __init__(self, EVENT=None, PID=None, E=None, px=None, py=None, pz=None, m=None):
        """
        Initialise this class with each data type.
        """
        self.EVENT = EVENT
        self.name = name
        self.PID = PID 
        self.E = E
        self.px = px
        self.py = py
        self.pz = pz
        self.m = m
        if self.PID == 2212: self.name = 'Proton'
        elif self.PID == -2212: self.name = 'Anti-Proton'
        elif self.PID == 2112: self.name = 'Neutron'
        elif self.PID == -2112: self.name = 'Anti-Neutron'
        
    def __str__(self):
        """
        Return a string to print of a Particle's data
        """
        return 'EVENT: {}, PARTICLE: {}, PID = {}, E = {}, P_x = {}, P_y = {}, P_z = {}, mass ={}'.\
            format(self.EVENT, self.name, self.PID, self.E, self.px, self.py, self.pz, self.m)
            
            
    def __repr__(self):
        return self.__class__.__name__+'EVENT: {}, PARTICLE: {}, PID = {}, E = {}, P_x = {}, P_y = {},\
            P_z = {}, mass = {}'.format(self.EVENT, self.name, self.PID, self.E, self.px, self.py, self.pz, self.m)
        

class Deuteron_DataBase(dict):
    
    def __init__(self, filename = "deuteron.dat"):
        data = open("deuteron.dat")
        par_str = ''
        for line in data:
            if line.startswith("#"):
                pass
            line = line.strip()
    
    
    def AddParticle(self, par_str):
        
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    