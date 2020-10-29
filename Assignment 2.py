# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class ParticleData:
    
    def __init__(self, pid=None, name=None, mass=None, tau=None, spin=None, charge=None, colour=None):
        self.pid = pid
        self.name = name
        self.mass = mass
        self.tau = tau
        self.spin = spin
        self.charge = charge
        self.colour = colour
        
    def __str__(self):
        return "{}, '{}', {}, {}, {}, {}, {}".format(self.pid,\
                    self.name, self.mass, self.tau, self.spin, self.charge, self.colour)
        
        
    def __repr__(self):
        return self.__class__.__name__+"({}, '{}', {}, {}, {}, {}, {})".format(self.pid,\
                    self.name, self.mass, self.tau, self.spin, self.charge, self.colour)
            
class ParticleDatabase(dict):
    
    def __init__(self, filename = "ParticleData.xml"):
        """
        
        """
        pstr = ""
        xml = open("ParticleData.xml")
        pstrlist = []
        
        for line in xml:
            line = line.strip()
            if line.startswith("<particle"): pstr = line
            elif pstr and line.endswith(">"):
               self.add(pstr + " " + line)
               pstr = ""
            pstrlist.append(pstr)
        xml.close()
        
    