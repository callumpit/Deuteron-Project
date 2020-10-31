# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from ps1 import Vector
from ps1 import Matrix
from ps1 import FourVector
import Assignment_2
from math import sqrt
from math import pi
from math import sin
from math import cos
import random

class ParticleData:
    """
    This Class represents the data for each particle type.
    """
    
    def __init__(self, pid=None, name=None, mass=None, tau=None, spin=None, charge=None, colour=None):
        """
        Constructor takes each data type as arguments and initialises.
        """
        self.pid = pid
        self.name = name  
        self.mass = mass
        self.tau = tau       # Define members of class
        self.spin = spin
        self.charge = charge
        self.colour = colour
        
    def __str__(self):
        """
        Return string to print of this Particle's data'
        """
        return "pid = {}, name = '{}',  mass = {}, tau = {}, spin = {}, charge = {}, colour = {}".format(self.pid,\
                    self.name, self.mass, self.tau, self.spin, self.charge, self.colour)
        
        
    def __repr__(self):
        """
        Return representation of particle's data'
        """
        return self.__class__.__name__+"({}, '{}', {}, {}, {}, {}, {})".format(self.pid,\
                    self.name, self.mass, self.tau, self.spin, self.charge, self.colour)
            
class ParticleDatabase(dict):
    
    def __init__(self, filename = "ParticleData.xml"):
        """
        Create Database to read in Pythia 8 particle data.
        """
        par_str = ""
        xml = open("ParticleData.xml")
        par_strlist = []
        # Loop through the file
        for line in xml:
            line = line.strip()
            if line.startswith("<particle"): par_str = line
            elif par_str and line.endswith(">"):
               self.add_data(par_str + " " + line)
               par_str = ""
            par_strlist.append(par_str)
        xml.close()
        
    def add_data(self, par_str):
        """
        Add XML data for each particle to the database.
        """
        import shlex
        # Make default dictionary
        par_dict = {"id": 0, "name": "", "antiName": None, "spinType": 0, \
                    "chargeType": 0, "colType": 0, "m0": 0, "tau0": 0}
        # split the string and loop through dictionary members.
        for pair in shlex.split(par_str[9:-1]):
            key, val = pair.split("=", 1)
            par_dict[key] = val
            
        # Add particle to the database.
        par_data = ParticleData( int(par_dict["id"]), par_dict["name"], float(par_dict["m0"]),\
              float(par_dict["tau0"]), int(par_dict["spinType"]), int(par_dict["chargeType"]), int(par_dict["colType"]))
        self[par_data.pid] = par_data
        self[par_data.name] = par_data
           
        # Check for anti-particle and if it exists change sign of pid and charge
        if par_dict["antiName"]:
            anti_data = ParticleData( -int(par_dict["id"]), par_dict["antiName"], float(par_dict["m0"]), \
                float(par_dict["tau0"]), int(par_dict["spinType"]), -1*int(par_dict["chargeType"]), int(par_dict["colType"]))
            self[anti_data.pid] = anti_data
            self[anti_data.name] = anti_data
            par_data.anti = anti_data
            anti_data.anti = par_data
            
###############################################################################
# Problem 3.
###############################################################################
            
class DiracMatrices(FourVector):
    """
    This class represents the Dirac Matrices
    """
    
    
    
    def __init__(self, v0=None, v1=None, v2=None, v3=None):
        """
        Initialise this class as a Fourvector with components being each Dirac
        Matrix.
        """       
        # Define each of the Dirac Matrices
        g0 = Matrix([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        g1 = Matrix([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
        g2 = Matrix([0.0, 0.0, 0.0, -1.0j], [0.0, 0.0, 1.0j, 0.0], [0.0, 1.0j, 0.0, 0.0], [-1.0j, 0.0, 0.0, 0.0])
        g3 = Matrix([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        FourVector.__init__(self, g0, g1, g2, g3) # Construct  Fourvector 
        
    # def __len__(self):
    #     return 4
        



###############################################################################
# Problem 4.
###############################################################################
class Particle:
    """
    This class represents a particle.
    """
    
    def __init__(self, data, p, h):
        """
        Initialise particle class with data, momentum and helicity
        """
        self.data = data
        self.p = p    # Define members of this class
        self.h = h
        if self.p[0] < 0: # Case for particle with negative energy
            self.p[0] = (self.p[1]**2 + self.p[2]**2 + self.p[3]**3 + \
            (Assignment_2.ParticleDatabase()[self.data.pid]).mass**2)**0.5
        
    

    # @classmethod
    def w(self):
        """
        Return the Dirac spinor for this particle.
        """
        
        if self.data.spin != 2: return None
        # Check if particle or anti-particle.
        s = -1 if self.data.pid < 0 else 1
        p = sqrt(sum([pj**2 for pj in self.p[1:]]))
        # Handle if |p| == p[3].
        
        if p + self.p[3] == 0:
            xi = 1.0
            if s*self.h == 1: kappa = [0, 1]
            elif s*self.h == -1: kappa = [-1, 0]
            else: kappa = [0, 0]
        # Handle otherwise.
        
        else:
            xi = 1.0/sqrt(2.0*p*(p + self.p[3]))
            if s*self.h == 1:
                kappa = [p + self.p[3], self.p[2]*1.0j + self.p[1]]
            elif s*self.h == -1:
                kappa = [self.p[2]*1.0j - self.p[1], p + self.p[3]]
            else:
                kappa = [0, 0]
                
        hp = xi*sqrt(self.p[0] + self.h*p)
        hm = xi*sqrt(self.p[0] - self.h*p)
        # Return the anti-particle spinor.
        
        if s == -1:
            return Vector(-self.h*kappa[0]*hp, -self.h*kappa[1]*hp,
                           self.h*kappa[0]*hm,  self.h*kappa[1]*hm)
        # Return the particle spinor.
        
        else:
            return Vector(kappa[0]*hm, kappa[1]*hm, kappa[0]*hp, kappa[1]*hp)
        
    def wbar(self):
        """
        Return the anti-particle Dirac spinor for this particle.
        """
        u = +self.w()
        return ~u*DiracMatrices()[0] # Define anti-particle Spinor
    
    
class Integrator:
    """
    This class utilises the Monte Carlo method of numerical integration to
    integrate a function between x and y limits.
    """
    
    def __init__(self, function, xmin, xmax, ymin, ymax):
        """
        Initialise the Integrator class with the integrand function and the
        x & y limits.
        """        
        self.function = function
        self.xmax = xmax    # Define members of cass
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.xrange = xmax - xmin
        self.yrange = ymax - ymin
     
    def mc(self, n=1000):
        """
        This method defines the Monte Carlo method of numerical integration, 
        a default number of sampling points of n=1000 is used.
        """
        r_points = []
        for i in range(n):
            point = []
            inside = []
            x = random.uniform(self.xmin, self.xmax) #Random x value
            y = random.uniform(self.ymin, self.ymax) #Random y value
            point.append(x)
            point.append(y)
            r_points.append(point)
        # print(r_points)
        # print(len(r_points))
        for i in range(n):
            func_val = self.function(r_points[i][0], r_points[i][1])
            # print(func_val)
            if abs(r_points[i][1])<=abs(func_val):
                inside.append(r_points[i])
        # print(len(inside))
        area = self.xrange*self.yrange
        proportional_area = (len(inside)/n)*area # Find proportion inside area
        return proportional_area
        
        
        
class Annihilate:
    
    def __init__(self, p1, p2, p3, p4):
        """
        Initialise this class with four members corresponding to the four 
        particles of the annihilation process e- e+ --> mu- mu+ . Constructor 
        takes these four particles as arguments.
        """

        self.p1 = p1  # electron
        self.p2 = p2  # positron
        self.p3 = p3  # muon
        self.p4 = p4  # anti-muon
        
        a = Vector(1,0,0,0)  # Construct Minkowski Matrix
        b = Vector(0,-1,0,0)
        c = Vector(0,0,-1,0)
        d = Vector(0,0,0,-1)
        Minkowski = Matrix(a,b,c,d)
        
        
        lower = +DiracMatrices()
        for i in range(0,3):       # Lowered index Dirac Matrices
            u = Minkowski*lower[i]
            lower[i] = u
            
        self.dmu = DiracMatrices()  # Raised index Dirac Matrices
        self.dml = lower
        
        self.q = sqrt(self.p1.p[0]**2 - self.p3.data.mass**2) # Find q for muons
        
        # print('dmu values:-------')
        # print(DiracMatrices())
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print('dml values:-------')
        # print(lower)
        
    def me(self):
        """
        This method returns the matrix element for the stored particles.
        """
        alpha = 1/137
        
        
        u = self.p1.w()  
        vbar = self.p2.wbar() # Construct Dirac Spinors 
        ubar = self.p3.wbar()
        v = self.p4.w()
        
        sum = 0
        
        coeff = ((-4*pi*alpha)/abs(self.p1.p + self.p2.p)**2)
        
        for mu in range(0,4):
            ME = coeff*(ubar*self.dmu[mu]*v)*(vbar*self.dml[mu]*u)
            sum += ME
        # print(sum)
        return sum   
            
            
        # # print(sum1)
         
        # for mu in range(0,4):
        #     x = (vbar*self.dml[mu]*u)
        #     sum2 += x
        # # print(sum2)
        
        # ME = coeff*sum1*sum2
        # # print('ME: {}'.format(ME))
        
        # return ME
    
    def xs(self, phi, theta):
        """
        This method returns the differential cross section of annihilation process
        for a given theta and phi.
        """
        if self.p3.data.pid == self.p4.data.pid:
            S = 0.5
        else:
            S = 1
            
        # Calculate momenta of muons  
        self.p3.p = FourVector(self.p1.p[0], self.q*sin(theta)*cos(phi), \
                     self.q*sin(theta)*sin(phi), self.q*cos(theta))
        self.p4.p = FourVector(self.p1.p[0], -self.q*sin(theta)*cos(phi), \
                     -self.q*sin(theta)*sin(phi), -self.q*cos(theta))
            
        hbar = 6.5821E-15
        c = 2.9979E+8
        coeff = ((hbar*c)/(8*pi))*S/(self.p1.p[0]+self.p2.p[0])**2
        # print('coeff: {}'.format(coeff))
        
        sum1list = []
        for i in range(1,4): sum1list.append(self.p3.p[i]**2)
        # print('sum1list: {}'.format(sum1list))
        sum_1 = 0
        for i in sum1list:
            sum_1 += i
        # print(print('sum_1: {}'.format(sum_1)))
        
        sum2list = []
        for i in range(1,4): sum2list.append(self.p1.p[i]**2)
        # print('sum2list: {}'.format(sum2list))
        sum_2 = 0
        for i in sum2list:
            sum_2 += i
            # print(sum_2)
        # print('sum_2: {}'.format(sum_2))
        
        dsigma = coeff*sqrt(sum_1/sum_2)*(self.me().conjugate())*self.me()*sin(theta)
        # print('dsigma: {}'.format(dsigma))
        return dsigma

        
        
        
def circle(x, y):
    """
    This general function takes the arguments x and y and returns 1 if they
    are within the area of the unit circle and 0 if not. 
    """
    if (x**2 + y**2)<1: # If inside radius
        return 1
    else: return 0
        
        
        
        

        
        
if __name__== "__main__":
    PDB = Assignment_2.ParticleDatabase()
    
    a = Vector(1,0,0,0)  # Construct Minkowski Matrix
    b = Vector(0,-1,0,0)
    c = Vector(0,0,-1,0)
    d = Vector(0,0,0,-1)
    Minkowski = Matrix(a,b,c,d)    
    
    electron = Particle(PDB['e-'], FourVector(sqrt(PDB['e-'].mass**2 + 10000), 0, 0, 100), 1)
    
    positron = Particle(PDB['e-'], FourVector(sqrt(PDB['e+'].mass**2 + 10000), 0, 0, -100), 1) 
    
    q = sqrt(electron.p[0]**2 - PDB['mu-'].mass**2)
    
    muon = Particle(PDB['mu-'], FourVector(sqrt(PDB['e-'].mass**2 + 10000), 0, 0, 0), 1)
    
    antimuon = Particle(PDB['mu-'], FourVector(sqrt(PDB['e-'].mass**2 + 10000), 0, 0, 0), 1)
    
    # print('MATRIX ELEMENT:')
    process = Annihilate(electron, positron, muon, antimuon) 
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('DIFFERENTIAL CROSS SECTION:')
    # Annihilate(electron, positron, muon, antimuon).xs(pi, 2*pi)
    # dsigma = Annihilate(electron, positron, muon, antimuon).xs(pi, 2*pi)
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('TOTAL CROSS SECTION:')
    a = Integrator(process.xs, 0, 2*pi, 0, pi)
    print(a.mc())

    I = Matrix([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])
    def DE(self):
        for mu in range(0,4):
            x = (DiracMatrices()[mu]*self.p[mu] - self.data.mass*I)*electron.w()
            print(x)
        
   
        
        
        
        
        
                
             
            
        
    