# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:30:23 2020

@author: Callum Pitceathly 
"""

##############################################################################
#                   Coding Project
##############################################################################

##############################################################################
#                Goal 1
##############################################################################
from ps1 import Vector
from ps1 import FourVector
from ps1 import Matrix
from ps1 import BoostMatrix
from math import sqrt
class Deuteron:
    """
    This Class represents the Particle data for protons, neutrons, anti-protons
    and anti-neutrons. 
    """
    def __init__(self, EVENT=0, PID=0, fvector=None, m=0):
        """
        Initialise this class with each data type.
        """
        # self.EVENT = EVENT
        self.PID = int(PID)
        self.name = None
        self.fvector = FourVector(*fvector) # Use FourVector to represent 
        self.m = float(m)             # particle Energy and Momentum.
        if self.PID == 2212: self.name = 'Proton' # Generate particle names.
        elif self.PID == -2212: self.name = 'Anti-Proton'
        elif self.PID == 2112: self.name = 'Neutron'
        elif self.PID == -2112: self.name = 'Anti-Neutron'
        
        
        
    def __str__(self):
        """
        Return a string to print of a Particle's useful data.
        """
        return ' PARTICLE : {}, {}, MASS = {}'\
            .format(self.name, repr(self.fvector), self.m)
            
            
    def __repr__(self):
        """
        Return Representation of particle data. 
        """
        return '(PARTICLE : {}, {}, MASS = {})'.format\
            (self.name, repr(self.fvector), self.m)
    
class Event_Database:
    
    def __init__(self):
        self.prev_event = 0
        self.m_d = 1.878 
        # self.deu_mass = 
        
    
    def read_event(self, number):
        data = open("deuteron.dat")
        event_number = 0
        event = []
        for line in data:
            line.strip()
            if line.startswith("#"):
                pass
            elif line == "\n":
                event_number += 1
                if event_number == number + 1:
                        # print(events)
                        event.insert(0, 'EVENT NUMBER: [{}]'.format(event_number - 1))
                        return event
                        # print(events)
                        # print('>>>')
                        break
                        
            else:
                if event_number == number:
                    List = line.split()
                    EP = FourVector(float(List[1]), float(List[2]), \
                                    float(List[3]), float(List[4]))
                    particle = Deuteron(event_number+1, int(List[0]), EP, List[5])
                    event.append(particle)
                    self.prev_event = event_number 
    
    def read_events(self, start, stop):
        self.prev_event = stop 
        events = []
        for i in range(start, stop+1):
            events.append(self.read_event(i))
        return events
    
    def next_event(self):
        return self.read_event(self.prev_event + 1)
        
    def combination(self, ev_num):
        p_pair = []
        anti_pair = []
        event = self.read_event(ev_num)
        for i in range(1, len(event)):
            for j in range(1, len(event)):
                if event[i].name=='Proton' and event[j].name=='Neutron': 
                    pair = [event[i], event[j]]
                    p_pair.append(pair)
                    
                if event[i].name=='Anti-Proton' and event[j].name=='Anti-Neutron':
                    anti = [event[i], event[j]]
                    anti_pair.append(anti) 
        return [*p_pair, *anti_pair]
    
    def k(self, pair):
        proton = pair[0].fvector
        neutron = pair[1].fvector
        BM = BoostMatrix(proton+neutron)
        # Create Boost Matrix for boosting a particle into the centre of mass
        # frame of the proton-neutron combination.
        boost_p = BM*proton  # Boost the proton 
        boost_n = BM*neutron # Boost the neutron
        print(repr(boost_p+boost_n))
        q = boost_p - boost_n # Calculate q
        print(q)
        return sqrt(sum([q[i]*q[i] for i in range(1,4)]))
    
    def 2_decay(self, pair):
        fvector = 
        
        
        
        
        
    
                
                
    

    # def __init__(self, filename = "short.dat"):
    #     data = open("short.dat")
    #     event_number = -1
    #     self.events = []
    #     event = []
    #     index = 0
    #     for line in data:
    #         line.strip()
    #         if line.startswith("#"):
    #             print('#')
    #         elif line == "\n":

    #             event_number += 1
    #             if event_number>0:
    #                     # print(events)
    #                     event.insert(0, [event_number])
    #                     self.events.append(event)
    #                     # print(events)
    #                     # print('>>>')
    #                     event = []
    #                     index = 0
    #         else:
    #             event.append(index)
    #             # print(event)
    #             # print('-----')
    #             index += 1
                
    #             # List = line.split()
    #             # EP = FourVector(float(List[1]), float(List[2]), float(List[3]), float(List[4]))
    #             # particle = Deuteron(self.event_number, int(List[0]), EP, List[5])
    #             # particle = Event_Database.CreateParticle(line)
    #             # Deuteron(particle.line_data)
    #             # event.append(particle)
    #             # print(event)
       
    #     # print(events)
    #     # return print(events)
            
    #     data.close()
        
    
        
                    # print(event)
                    # print('-----')
                    
        
        
        
            
        
         
        
    # def CreateParticle(self, line):
    #     self.line = line
    #     List = self.line.split()
        
    #     self.event = Event_Database.event_number
    #     self.PID = int(List[0])
    #     self.fvector = FourVector(float(List[1]), float(List[2]), float(List[3]), float(List[4]))
    #     self.m = float(List[5])
    #     self.line_data = [self.event, self.PID, self.fvector, self.m] 
        
    # def AddParticle(self, particle):
    #     dct = 
        
            
    
    
# data = open("short.dat") 
# for line in data:
#     line = line.strip()
#     if line.startswith("#"):
#         pass
#     elif line == "":
#         pass
#     else: print(line.split())
        
#     # data_cont = data.readline()
#     # print(data_cont)
    # line = line.strip()
# data_cont = data.readline()
# print(data_cont)

# data_cont = data.readline()
# print(data_cont)

# data_cont = data.readline()
# print(data_cont)
# data.close()     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        