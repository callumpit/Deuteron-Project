# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:43:03 2020

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
from math import pi
from math import cos
from math import sin
from math import exp
import random

class Particle:
    """
    This Class represents the Particle data for protons, neutrons, anti-protons,
    anti-neutrons, deuterons and anti-deuterons. 
    """
    def __init__(self, EVENT=0, PID=0, fvector=None, m=0):
        """
        Initialise this class with each data type.
        """
        # self.EVENT = EVENT
        self.PID = int(PID)  # Particle ID.
        self.name = None  # Particle name.
        self.fvector = FourVector(*fvector) # Use FourVector to represent 
                                       #  # particle Energy and Momentum.
        self.m = float(m)   # Particle mass.         
        if self.PID == 2212: self.name = 'Proton' # Generate particle names.
        elif self.PID == -2212: self.name = 'Anti-Proton'
        elif self.PID == 2112: self.name = 'Neutron'
        elif self.PID == -2112: self.name = 'Anti-Neutron'
        elif self.PID == 700201: self.name = 'Deuteron'
        elif self.PID == -700201: self.name = 'Anti-Deuteron'
         
        
        
        
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
    
class Event_Data:
    """
    This Class contains methods for analysing the data in the "deuteron.dat" file.
    """
    
    def __init__(self):
        """
        This method stores useful values and sets event numbers
        to zero for use later on in methods of the class.
        """
        self.event_number = 0 # Event number and previous event number set to zero for later methods
        self.prev_event = 0
        self.m_d = 1.878      # Mass data for useful particles
        self.m_pion = 0.1396
        self.m_pion0 = 0.135
        # self.deu_mass = 
        
    
    def read_event(self, number):
        """
        This method takes the event number as an argument and returns all 
        particle data for said event.
        """
        data = open("deuteron.dat") # Open data file
        self.event_number = 0 
        event = [] # Create empty list for event
        for line in data:
            line.strip() # Remove whitespace for each line
            if line.startswith("#"): # Ignore first line 
                pass
            elif line == "\n": # For empty lines corresponding to event ending and new one being read in
                self.event_number += 1 # Set event number for next event
                if self.event_number == number + 1: # Check previous event being read is the desired event.
                        # print(events)
                        event.insert(0, 'EVENT NUMBER: [{}]'.format(self.event_number - 1)) # Put event label at start for list
                        return event # Return event
                        # print(events)
                        # print('>>>')
                        # break
                        
            else: # For lines which contain particle data
                if self.event_number == number: # Checks event being read is desired event
                    List = line.split() # Create list of particle data
                    EP = FourVector(float(List[1]), float(List[2]), \
                                    float(List[3]), float(List[4])) # Create fourvector of particle data
                    particle = Particle(self.event_number+1, int(List[0]), EP, List[5]) # Create particle
                    event.append(particle) # Add particle to event list
                    self.prev_event = self.event_number # Set previous event value to the event just read. 
        data.close() # Close data file
    
    def read_events(self, start, stop):
        """
        This method takes a start and stop event number and returns all 
        particle data for these events and all events between with each seperate
        event being numbered and labelled.
        """
        self.prev_event = stop # Set previous event value to the last event read in by method.
        events = [] # Create empty list for events
        for i in range(start, stop+1):
            events.append(self.read_event(i)) # Read in each event
        return events # Return list of events
    
    def next_event(self):
        """
        Returns all particle data for the event following the last event that 
        has been read in.
        """
        return self.read_event(self.prev_event + 1)
        
    def combination(self, event, pn=False):
        """
        Takes an event list as an argument and returns all possible particle
        combinations for said event. If pn is set to True, only proton-neutron
        or anti-proton - anti-neutron combinations are considered.
        """
        p_pair = [] # Create empty list for p-n pairs
        anti_pair = [] # Create empty list for anti p-n pairs
        
        if pn == True:
            for i in event: # Search event for different combinations
                for j in event:
                    if i.name=='Proton' and j.name=='Neutron': # Condition for p-n combinations
                        pair = [i, j]
                        p_pair.append(pair)
                        
                    if i.name=='Anti-Proton' and j.name=='Anti-Neutron': # Condition for anti p-n combinations
                        anti = [i, j]
                        anti_pair.append(anti)
            combinations = [*p_pair, *anti_pair]
        else:
            for i in event: # Create all possible p-n and anti combinations
                for j in event:
                    
                    # Create all normal combinations.
                    if i.PID > 0 and j.PID > 0:
                        if i.fvector != j.fvector:
                            pair = [i, j]
                            p_pair.append(pair)
                    
                    # Create all anti combinations
                    if i.PID < 0 and j.PID < 0:
                        if i.fvector != j.fvector:
                            anti = [i, j]
                            anti_pair.append(anti)
                          
            combinations = [*p_pair, *anti_pair] # Create list of all possible combinations
            p_pair, anti_pair = [], [] # Set lists back to empty for next event
            event = [] # Set event list back to empty for next event
            # Remove identical combinations due to double counting
            for i in combinations:
                for j in combinations:
                    if i == [j[1], j[0]]:
                        combinations.remove(j)
            
        return combinations # Return list of particle and anti-particle combinations
    
    @staticmethod
    def k(pair):
        """
        Takes a particle pair as an argument and returns the k value for this 
        pair.
        """
        proton, neutron = pair[0].fvector, pair[1].fvector # Aquire particle momenta
        M = pair[0].m + pair[1].m
        try: BM = BoostMatrix(proton+neutron)
        except: BM = BoostMatrix(proton+neutron, M)    
        # Create Boost Matrix for boosting a particle into the centre of mass
        # frame of the proton-neutron combination.
        boost_p, boost_n = BM*proton, BM*neutron  # Boost the proton and neutron 
        # print(repr(boost_p+boost_n))
        q = boost_p - boost_n # Calculate q
        # print(q)
        return sqrt(sum([q[i]*q[i] for i in range(1,4)]))
    
    def two_decay(self, pair, product):
        """
        Takes a p-n (or anti) pair and the lighter product of this decay as
        arguments and returns particle data for both products boosted back into
        the lab frame.
        """
        # Determine combined mass of p1 and p2
        m = pair[0].m + pair[1].m
        
        # Create particle Four Vectors.
        p1, p2 = pair[0].fvector, pair[1].fvector 
        # Determine decay products and masses.
        m2 = self.m_d
        if product == 'Photon': m1 = 0
        elif product == 'Pion': m1 = self.m_pion
        elif product == 'Pion0': m1 = self.m_pion0
        if pair[0].PID > 0: d_PID = 700201
        if pair[0].PID < 0: d_PID = -700201
        try: BM = BoostMatrix(p1+p2) # Create Boost Matrix for boosting a particle 
        except: BM = BoostMatrix(p1+p2, m)# into the centre of mass frame of the particle combination.
        
        boost_p1, boost_p2 = BM*p1, BM*p2  # Boost the particles into COM.
        fvector = boost_p1 + boost_p2 # Create FourVector of boosted combination.
        M = fvector[0] # Define centre of mass energy.
        
        # Calculate the magnitude of both product's momenta.
        mag_p = sqrt(((M**2 - (m1 + m2)**2))*(M**2 - (m1 - m2)**2))/(2*M)
        # print(mag_p)
        # Generate random phi and theta values.
        theta, phi = random.uniform(0, pi), random.uniform(0, 2*pi)
        # Calculate energy components of each product Four Vector.
        E1, E2 = (M**2 - m2**2 + m1**2)/(2*M), (M**2 - m1**2 + m2**2)/(2*M)
        # Calculate momentum components of each product Four Vector.
        P1 = FourVector(E1, -mag_p*sin(phi)*cos(theta),\
                        -mag_p*sin(phi)*sin(theta), -mag_p*cos(phi))
        P2 = FourVector(E2, mag_p*sin(phi)*cos(theta),\
                        mag_p*sin(phi)*sin(theta), mag_p*cos(phi))
        try: BM = BoostMatrix(~(p1+p2))
        except: BM = BoostMatrix(~(p1+p2), m)
        # P1, P2 = BM*P1, BM*P2  # This line has been changed for the dal_raklev method
        P2 = BM*P2
        
        return [P1, Particle(self.event_number, d_PID, P2, self.m_d)]
    
    def three_decay(self, pair, product1, product2):
        """
        Takes particle pair and the two lighter products of three body decay 
        and returns particle data for all products boosted back into the lab 
        frame.
        """
        # Determine deuteron type
        if pair[0].PID > 0: d_PID = 700201
        if pair[0].PID < 0: d_PID = -700201
        
        # Determine combined mass of combination
        M = pair[0].m + pair[1].m
        
        # Calculate combined FourVector of particle pair
        p01, p02 = pair[0].fvector, pair[1].fvector
        p0 = (p01 + p02)
        
        # Generate boost matrices
        try: BM = BoostMatrix(p0) # Create BoostMatrix for COM frame
        except: BM = BoostMatrix(p0, M)
        try: rev_BM = BoostMatrix(~p0) # Create BoostMatrix to return to lab frame
        except: rev_BM = BoostMatrix(~p0, M)
        
        # Boost particles into COM frame
        p01_boost, p02_boost = BM*p01, BM*p02
        p0 = p01_boost + p02_boost
        m0 = p0[0] # Calculate COM energy
        
        # Determine masses of products
        m3 = self.m_d
        if product1 == 'Pion': m1 = self.m_pion
        elif product1 == 'Pion0': m1 = self.m_pion0
        if product2 == 'Pion': m2 = self.m_pion
        elif product2 == 'Pion0': m2 = self.m_pion0
        mSum = m1 + m2 + m3
        mDiff = m0 - mSum # Mass difference between COM energy and product masses
        
        # Calculate minimum momentum for intermediate decay products.
        m23Min = m2 + m3
        
        # Choose intermediate mass randomly in allowed range
        m23 = m23Min + random.random() * mDiff
        
        # Calculate magnitude of Momenta of particle 1 and intermidiate 
        # combined mass particle 23.
        p1Abs  = 0.5 * sqrt( (m0 - m1 - m23) * (m0 + m1 + m23) \
        * (m0 + m1 - m23) * (m0 - m1 + m23) ) / m0; 
        p23Abs = 0.5 * sqrt( (m23 - m2 - m3) * (m23 + m2 + m3) \
          * (m23 + m2 - m3) * (m23 - m2 + m3) ) / m23
            
        # Set up decay of intermediate particle m23 -> m2 + m3 in its rest frame
        cosTheta = 2 * random.random() - 1 # Define random cosTheta
        sinTheta = sqrt(1 - cosTheta*cosTheta) # Define sinTheta
        phi = 2 * pi * random.random()  # Define random phi value 
        px = p23Abs * sinTheta * cos(phi) 
        py = p23Abs * sinTheta * sin(phi)  # Determine momentum components
        pz = p23Abs * cosTheta
        E2 = sqrt(m2*m2 + p23Abs*p23Abs) # Determine energy of products
        E3 = sqrt(m3*m3 + p23Abs*p23Abs)
        prod2 = FourVector(E2, px, py, pz) # Create product FourVectors
        prod3 = FourVector(E3, -px, -py, -pz)
        
        # Set up initial decay m0 -> m1 + m23 in its rest frame    
        cosTheta = 2 * random.random() - 1 # Define random cosTheta
        sinTheta = sqrt(1 - cosTheta*cosTheta) # Define sinTheta
        phi = 2 * pi * random.random() # Define random phi value
        px              = p1Abs * sinTheta * cos(phi) 
        py              = p1Abs * sinTheta * sin(phi)  # Determine momentum components
        pz              = p1Abs * cosTheta
        E1 = sqrt(m1*m1 + p1Abs*p1Abs) # Determine energy of products
        
        prod1 = FourVector(E1, px, py, pz) # product 1 FourVector
        # Boost back into lab frame
        prod1, prod2, prod3 = rev_BM*prod1, rev_BM*prod2, rev_BM*prod3
        
        return [prod1, prod2, Particle(self.event_number, d_PID, prod3, self.m_d)]
    
class cross_sec:
    """
    This class defines methods for calculating the cross sections for processes
    in the Dal Raklev algorithm in micro barns
    """
    def __init__(self):
        self.m_pion = 0.1396
        
    def pp_two_body(self, q):
        """
        Takes argument q and returns corresponding cross-section for process
        "p + n -> pion0 + pion0 + d" and conjugate process. q is the momentum 
        of the outgoing pion in the COM frame.
        """
        q = float('%.8g' % q)
        a, b, c, d, e = 170, 1.34, 1.77, 0.38, 0.096 # Define best fit value parameters
        n = q / self.m_pion
        
        try: return float('%.8g' % ((a * n**b)/((c - exp(d*n))**2 + e)))           
        except: return 0
            
    
    def nn_two_body(self, q):
        """
        Takes argument q and returns corresponding cross-section for process
        "p + n -> pion0 + pion0 + d" and conjugate process. q is the momentum
        of the outgoing pion in the COM frame.
        """
        q = float('%.8g' % q)
        a, b, c, d, e = 170, 1.34, 1.77, 0.38, 0.096 # Define best fit value parameters
        n = q / self.m_pion
        
        try: return float('%.8g' % ((a * n**b)/((c - exp(d*n))**2 + e)))            
        except: return 0
            
    
    def pn_two_body(self, q):
        """
        Takes argument q and returns corresponding cross-section for process
        "p + n -> pion0 + pion0 + d" and conjugate process. q is the momentum
        of the outgoing pion in the COM frame.
        """
        q = float('%.8g' % q)
        a, b, c, d, e = 170, 1.34, 1.77, 0.38, 0.096 # Define best fit value parameters
        n = q / self.m_pion
        
        try: return float('%.8g' % ((a * n**b)/(2 * ((c - exp(d*n))**2 + e))))
        except: return 0
            
    
    def pp_three_body(self, k):
        """
        Takes argument k and returns corresponding cross-section for process
        "p + p -> pion + pion0 + d" and conjugate process.
        """
        k = float('%.8g' % k)   # Define best fit value parameters
        a, b, c, d, e = 5.099E+15, 16.56, 2.333E+7, 13.33, 2.868E+16
        
        try: return float('%.8g' % ((a * k**b)/((c - exp(d*k))**2 + e)))
        except: return 0    
                
            
        
            
    
    def nn_three_body(self, k):
        """
        Takes argument k and returns corresponding cross-section for process
        "n + n -> pion + pion0 + d" and conjugate process.
        """
        k = float('%.8g' % k)   # Define best fit value parameters
        a, b, c, d, e = 5.099E+15, 16.56, 2.333E+7, 13.33, 2.868E+16
        
        try: return float('%.8g' % ((a * k**b)/((c - exp(d*k))**2 + e)))
        except: return 0 
                
            
        
            
    
    def pn_three_body_0(self, k):
        """
        Takes argument k and returns corresponding cross-section for process
        "p + n -> pion0 + pion0 + d" and conjugate process.
        """
        k = float('%.8g' % k)  # Define best fit value parameters
        a, b, c, d, e = 2.855E+6, 13.11, 2.961E+3, 5.572, 1.461E+6
        
        try: return float('%.8g' % ((a * k**b)/((c - exp(d*k))**2 + e)))
        except: return 0
                
            
        
            
    
    def pn_three_body_1(self, k):
        """
        Takes argument k and returns cross section for a given k.
        For process "p + n -> pion + pion + d" and its conjugate process.
        """
        k = float('%.8g' % k)  # Define best fit value parameters
        a1, b1, c1, d1, e1 = 6.465E+6, 10.51, 1.979E+3, 5.363, 6.045E+5
        a2, b2, c2, d2, e2 = 2.549E+15, 16.57, 2.33E+7, 11.19, 2.868E+16
        
        try:  # Calculate fractions involved in equation
            frac1 = float('%.8g' % ((a1 * k**b1)/((c1 - exp(d1*k))**2 + e1)))
            frac2 = float('%.8g' % ((a2 * k**b2)/((c2 - exp(d2*k))**2 + e2)))            
            return frac1 + frac2
            
        except:
            return 0
    
    def pn_photon(self, k):
        """
        Takes argument k and returns cross section for a given k.
        For process "p + n -> photon + d" and its conjugate process.
        """
        # Define best fit value parameters
        coef = [2.30346, -93.66346, 2.565390E+3, -2.5594101E+4, 1.43513109E+5, \
                -5.0357289E+5, 1.14924802E+6, -1.72368391E+6, 1.67934876E+6, \
                   -1.01988855E+6, 3.4984035E+5, -5.1662760E+4]
        b1, b2 = -5.1885, 2.9196
        k = float('%.8g' % k)
        if k < 1.28:
            summed = coef[0]*(k**-1) + coef[1]*(k**0) + coef[2]*(k**1) + \
                    coef[3]*(k**2) + coef[4]*(k**3) + coef[5]*(k**4) + coef[6]*(k**5) \
                     + coef[7]*(k**6) + coef[8]*(k**7) + coef[9]*(k**8) + coef[10]*(k**9) \
                     + coef[11]*(k**10)
            
            return summed*1E-6
            
                
        elif k >= 1.28: return exp(-b1*k - b2*k**2)*1E-6
            
            
        
    
def coalescence():
    """
    This function implements the coalescence model.

    Returns
    -------
    deuterons : List
        This list contains all deuterons produced from events simulated using
        the coalescence model.

    """
    deuterons = [] # Create an empty list for deuterons and anti-deuterons.
    a = Event_Data() # Create instance of event data class.
    data = open("deuteron.dat") # Read in the event data.
    a.event_number = 0 # Set event number equal to zero
    event = [] # Create empty list for each event

    for line in data:
        line.strip() # Remove whitespace in each line
        if line.startswith("#"): # Ignore first line 
            pass
        elif line == "\n": # For empty lines
            a.event_number += 1 # Each empty line means a new event follows
            
            combinations = a.combination(event, pn=True)
            event = [] # Set event list back to empty for next event
            
            if combinations == []: pass # Ignore events which do not give any combinations
            else:
                for i in combinations: # Calculate k for each combination
                    if len(i) < 3: # Only check combinations with unused nucleons.
                        k = a.k(i)
                        if k < 0.193:  # Cutoff condition for deuteron production
                            decay = a.two_decay(i, 'Photon') # Initiate decay for combinations below cut-off
                            deuterons.append(decay[1]) # Add deuteron fourvectors to deuteron list
                            
                            # Remove any combinations involving one or both of formation nucleons.
                            for u in combinations:
                                if i[0] == u[0] or i[1] == u[0]: u.append('Ignore') 
                                if i[0] == u[1] or i[1] == u[1]: u.append('Ignore')
                    else: pass   
                
        # Now extract particle data for all other lines               
        else:

            List = line.split() # Extract particle properties
            EP = FourVector(float(List[1]), float(List[2]), \
                            float(List[3]), float(List[4]))
            particle = Particle(a.event_number+1, int(List[0]), EP, List[5]) # Create particle data for each
            event.append(particle) # Add each particle to event list
            
    data.close()
    return deuterons
    

    
def dal_rak():
    """
    This function implements the Dal-Raklev model.

    Returns
    -------
    deuterons : List
        This list contains all deuterons and anti-deuterons produced in all
        events simulated using the Dal-Raklev model.

    """
    deuterons = [] # Create an empty list for deuterons and anti-deuterons.
    a = Event_Data() # Create instance of event data class.
    data = open("deuteron.dat") # Read in the event data.
    a.event_number = 0 # Set event number equal to zero
    event = [] # Create empty list for each event
    sigma_0 = (1/2.63)*1E+6 # Value from D-R paper used for 1/sigma_0 was (1/2.63)[b^-1]
    
    
    for line in data:
        line.strip() # Remove whitespace in each line
        if line.startswith("#"): # Ignore first line 
            pass
        elif line == "\n": # For empty lines
            a.event_number += 1 # Each empty line means a new event follows
            print('---------------------EVENT NUMBER {} -----------------'.format(a.event_number-1))
            
            # Now determine all possible particle combinations
            combinations = a.combination(event)
            
            if combinations == []: pass # Ignore events which do not give any combinations
            else:
                # Calculate k for each combination
                for i in combinations: 
                    if len(i) < 3:
                        k = a.k(i)
                    
                        # FOR PP COMBINATION
                        if (i[0].name == 'Proton' and i[1].name == 'Proton') or \
                            (i[0].name == 'Anti-Proton' and i[1].name == 'Anti-Proton'):
                            selects = []
                            
                            # Two body decay channel is checked
                            try:
                                vec = a.two_decay(i, 'Pion')[0]
                                q = sqrt(vec[1]*vec[1] + vec[2]*vec[2] + vec[3]*vec[3])
                                xs_two_body = cross_sec().pp_two_body(q)
                                prob_pp2 = xs_two_body/sigma_0
                                if random.random() < prob_pp2: selects.append('two_body')
                            except ValueError:
                                pass
                            
                            # Three body decay channel is checked
                            try:
                                vec3 = a.three_decay(i, 'Pion', 'Pion0')[2]
                                xs_three_body = cross_sec().pp_three_body(k)
                                prob_pp3 = xs_three_body/sigma_0
                                if random.random() < prob_pp3: selects.append('three_body')
                            except ValueError:
                                pass
                            
                            # Check number of channels selected and perform decay
                            if len(selects) == 0: pass
                            if len(selects) > 0:
                                if len(selects) == 1 and selects[0] == 'two_body':
                                    deuterons.append(a.two_decay(i, 'Pion')[1])
                                if len(selects) == 1 and selects[0] == 'three_body':
                                    deuterons.append(vec3)
                                if len(selects) == 2:
                                    norm_fac = xs_two_body + xs_three_body
                                    xs_two_body /= norm_fac
                                    xs_three_body /= norm_fac
                                    choice = random.choices(population=['two_body', 'three_body'],\
                                    weights=[xs_two_body, xs_three_body], k=1)
                                    if choice == 'two_body': deuterons.append(a.two_decay(i, 'Pion')[1])   
                                    if choice == 'three_body': deuterons.append(vec3)
                                    
                                # Ensure nucleons involved aren't used again.
                                for u in combinations:
                                    if i[0] == u[0] or i[1] == u[0]: u.append('Ignore') 
                                    if i[0] == u[1] or i[1] == u[1]: u.append('Ignore')
                                
                                
                        # FOR NN COMBINATION
                        if (i[0].name == 'Neutron' and i[1].name == 'Neutron') or \
                            (i[0].name == 'Anti-Neutron' and i[1].name == 'Anti-Neutron'):
                            selects = []
                            
                            # Two body decay channel is checked
                            try:
                                vec = a.two_decay(i, 'Pion')[0]
                                q = sqrt(vec[1]*vec[1] + vec[2]*vec[2] + vec[3]*vec[3])
                                xs_two_body = cross_sec().nn_two_body(q)
                                prob_nn2 = xs_two_body/sigma_0
                                if random.random() < prob_nn2: 
                                    selects.append('two_body')
                            except ValueError:
                                pass
                            
                            # Three body decay channel is checked
                            try:
                                vec3 = a.three_decay(i, 'Pion', 'Pion0')[2]
                                xs_three_body = cross_sec().nn_three_body(k)
                                prob_nn3 = xs_three_body/sigma_0
                                if random.random() < prob_nn3:
                                    selects.append('three_body')
                                    # print('NN3>>>>>>>>>>>>>>>')
                            except ValueError:
                                pass
                            
                            # Check number of channels selected and perform decay
                            if len(selects) == 0: pass
                            if len(selects) > 0:
                                if len(selects) == 1 and selects[0] == 'two_body':
                                    deuterons.append(a.two_decay(i, 'Pion')[1])
                                if len(selects) == 1 and selects[0] == 'three_body':
                                    deuterons.append(vec3)
                                if len(selects) == 2:
                                    norm_fac = xs_two_body + xs_three_body
                                    xs_two_body /= norm_fac
                                    xs_three_body /= norm_fac
                                    choice = random.choices(population=['two_body', 'three_body'],\
                                    weights=[xs_two_body, xs_three_body], k=1)
                                    if choice == 'two_body': deuterons.append(a.two_decay(i, 'Pion')[1])   
                                    if choice == 'three_body': deuterons.append(vec3)
                                    
                                    # Ensure nucleons involved aren't used again.
                                for u in combinations:
                                    if i[0] == u[0] or i[1] == u[0]: u.append('Ignore') 
                                    if i[0] == u[1] or i[1] == u[1]: u.append('Ignore')
                        
                        # FOR PN COMBINATION
                        if (i[0].name == 'Proton' and i[1].name == 'Neutron') or \
                            (i[0].name == 'Anti-Proton' and i[1].name == 'Anti-Neutron')\
                                or (i[0].name == 'Neutron' and i[1].name == 'Proton') or \
                            (i[0].name == 'Anti-Neutron' and i[1].name == 'Anti-Proton'):
                            selects = []
                            # print(i)
                                
                            # Two body decay channels are checked.                        
                            # First the Pion two body channel.
                            try:
                                vec = a.two_decay(i, 'Pion0')[0]
                                q = sqrt(vec[1]*vec[1] + vec[2]*vec[2] + vec[3]*vec[3])
                                xs_two_body = cross_sec().pn_two_body(q)                           
                                prob_pn2 = xs_two_body/sigma_0
                                if random.random() < prob_pn2: selects.append('two_body')
                            except ValueError:
                                pass
                            
                            # Now the photon production channel.
                            try:
                                vec_photon = a.two_decay(i, 'Photon')[1]
                                xs_photon = cross_sec().pn_photon(k)                            
                                prob_photon = xs_photon/sigma_0
                                if random.random() < prob_photon: selects.append('photon')
                            except ValueError:
                                pass
                            
                            # Three body decay channels are checked.
                            # First the Pion0 channel.
                            try:
                                vec30 = a.three_decay(i, 'Pion0', 'Pion0')[2]
                                xs_three_body0 = cross_sec().pn_three_body_0(k)                            
                                prob_pn30 = xs_three_body0/sigma_0
                                if random.random() < prob_pn30: selects.append('three_body0')
                            except ValueError:
                                pass
                            
                            # Now the Pion channel.
                            try:
                                vec31 = a.three_decay(i, 'Pion', 'Pion')[2]   
                                xs_three_body1 = cross_sec().pn_three_body_1(k)
                                prob_pn30 = xs_three_body1/sigma_0
                                if random.random() < prob_pn30: selects.append('three_body1')
                            except ValueError:
                                pass
                            
                            # Check number of channels selected and perform decay
                            if len(selects) == 0: pass
                            
                            if len(selects) > 0:
                                if len(selects) == 1 and selects[0] == 'two_body':
                                    deuterons.append(a.two_decay(i, 'Pion0')[1])
                                if len(selects) == 1 and selects[0] == 'photon':
                                    deuterons.append(vec_photon)
                                if len(selects) == 1 and selects[0] == 'three_body0':
                                    deuterons.append(vec30)
                                if len(selects) == 1 and selects[0] == 'three_body1':
                                    deuterons.append(vec31)
                                if len(selects) < 5:
                                    
                                    # When multiple channels are selected, 
                                    #perform weighted random choice.
                                    w = []
                                    for s in selects:
                                        if s == 'two_body': w.append(xs_two_body)
                                        if s == 'photon': w.append(xs_photon)
                                        if s == 'three_body0': w.append(xs_three_body0)
                                        if s == 'three_body1': w.append(xs_three_body1)
                                    norm_fac = sum([n for n in w])
                                    for m in range(0,len(w)): w[m] /= norm_fac
                                    choice = random.choices(population=selects, weights=w, k=1)
                                    if choice == 'two_body': deuterons.append(a.two_decay(i, 'Pion0')[1])
                                    if choice == 'photon': deuterons.append(vec_photon)
                                    if choice == 'three_body0':
                                        deuterons.append(vec30)
                                    if choice == 'three_body1':
                                        deuterons.append(vec31)
                                        
                                # Ensure nucleons involved aren't used again.
                                for u in combinations:
                                    if i[0] == u[0] or i[1] == u[0]: u.append('Ignore') 
                                    if i[0] == u[1] or i[1] == u[1]: u.append('Ignore')
                    
                    # Ignore combinations involving previously used nuclei.              
                    else: pass 
                
        # Now extract particle data for all other lines               
        else:

            List = line.split() # Extract particle properties
            EP = FourVector(float(List[1]), float(List[2]), \
                            float(List[3]), float(List[4]))
            particle = Particle(a.event_number+1, int(List[0]), EP, List[5]) # Create particle data for each
            event.append(particle) # Add each particle to event list
            
    data.close()
    return deuterons
    
    
    
# if __name__== "__main__":
#     from datetime import datetime
#     start_time = datetime.now()
#     # dr = dal_rak()
#     coal = coalescence()
#     end_time = datetime.now()
#     print('Duration: {}'.format(end_time - start_time))
#     a = Event_Data()
    
    
    
    
    
    
    
    
    
    
    
