# Import Problem Set 1 classes.
from ps1 import Vector, Matrix, FourVector

###############################################################################
# Problem 1.
###############################################################################
class ParticleData:
    """
    The 'ParticleData' class stores all the necessary information to
    define a particle.
    """
    ###########################################################################
    # Problem 1(a), 1(b), 1(c), 2(d).
    ###########################################################################
    def __init__(self, pid = None, name = None, mass = None, tau = None,
                 spin = None, charge = None, colour = None):
        """
        Initialize the class with the following: 'pid' is the particle ID
        number, 'name' the name, 'mass' the mass in GeV, 'tau' the
        proper lifetime in mm/c, 'spin' the particle spin, 'charge' is
        three times the electromagnetic charge, and 'colour' is the
        colour type.
        """
        self.pid = pid
        self.name = name
        self.mass = mass
        self.tau = tau
        self.spin = spin
        self.charge = charge
        self.colour = colour
        self.anti = None
    ###########################################################################
    # Problem 1(e).
    ###########################################################################
    def __str__(self):
        """
        Return a string to print of this particle data.
        """
        return ("%6s: %s\n"*6 + "%6s: %s") % (
            "pid", self.pid, "name", self.name, "mass", self.mass,
            "tau", self.tau, "spin", self.spin, "charge", self.charge,
            "colour", self.colour)
    def __repr__(self):
        """
        Return the representation of this particle data.
        """
        return "ParticleData(%r, %r, %r, %r, %r, %r, %r)" % (
            self.pid, self.name, self.mass, self.tau, self.spin,
            self.charge, self.colour)
    
###############################################################################
# Problem 2.
###############################################################################
class ParticleDatabase(dict):
    """
    The 'ParticleDatabase' initializes and stores the 'ParticleData' for
    all particle in the 'ParticleData.xml' file from Pythia 8.
    """
    ###########################################################################
    # Problem 2(a), 2(b).
    ###########################################################################
    def __init__(self, xmlfile = "ParticleData.xml"):
        """
        Read in the particle data from the XML file 'xmlfile'.
        """
        # Instantiate the base class.
        dict.__init__(self)
        # Open the XML file.
        xml = open(xmlfile)
        # Create the particle string.
        pstr = ""
        # Loop over the file.
        for line in xml:
            line = line.strip()
            if line.startswith("<particle"): pstr = line
            elif pstr and line.endswith(">"):
                self.add(pstr + " " + line)
                pstr = ""
        xml.close()
    ###########################################################################
    # Problem 2(c), 2(d).
    ###########################################################################
    def add(self, pstr):
        """
        Parses the XML for a particle and adds it to the database.
        """
        import shlex
        # Create the default dictionary.
        pdct = {"id": 0, "name": "", "antiName": None, "spinType": 0,
                "chargeType": 0, "colType": 0, "m0": 0, "tau0": 0}
        # Split the string by spaces, and loop over the entries.
        for pair in shlex.split(pstr[9:-1]):
            # Split each string into a key-value pair.
            key, val = pair.split("=", 1)
            pdct[key] = val
        # Add the particle.
        pdat = ParticleData(
            int(pdct["id"]), pdct["name"], float(pdct["m0"]),
            float(pdct["tau0"]), int(pdct["spinType"]),
            int(pdct["chargeType"]), int(pdct["colType"]))
        self[pdat.pid] = pdat
        self[pdat.name] = pdat
        # Add the anti-particle if it exists, flip PID and charge.
        if pdct["antiName"]:
            adat = ParticleData(
            -int(pdct["id"]), pdct["antiName"], float(pdct["m0"]),
            float(pdct["tau0"]), int(pdct["spinType"]),
            -1*int(pdct["chargeType"]), int(pdct["colType"]))
            self[adat.pid] = adat
            self[adat.name] = adat
            pdat.anti = adat
            adat.anti = pdat
            
###############################################################################
# Problem 3.
###############################################################################
class DiracMatrices(FourVector):
    """
    This class provides the Dirac matrices. Note that this class
    inherits from the 'FourVector' class. This is because the Dirac
    matrices also transform under the Minkowski metric, just like
    standard four-vectors.
    """
    ###########################################################################
    # Problem 3(a), 3(b).
    ###########################################################################
    def __init__(self, v0 = None, v1 = None, v2 = None, v3 = None):
        """
        Initialize the Dirac matrices. Ideally this would not be mutable.
        """
        g0 = Matrix([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        g1 = Matrix([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
        g2 = Matrix([0.0, 0.0, 0.0, -1.0j], [0.0, 0.0, 1.0j, 0.0],
                    [0.0, 1.0j, 0.0, 0.0], [-1.0j, 0.0, 0.0, 0.0])
        g3 = Matrix([0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        FourVector.__init__(self, g0, g1, g2, g3)

###############################################################################
# Problem 4.
###############################################################################
class Particle:
    """
    This class represents a particle.
    """
    ###########################################################################
    # Problem 4(a), 4(b).
    ###########################################################################
    def __init__(self, data, p, h):
        """
        Initialize the 'Particle' class, given 'data' of type
        'ParticleData' for that particle type, the momentum
        four-vector 'p', and the helicity 'h'.
        """
        from math import sqrt
        self.data = data
        self.p = +p
        if self.p[0] < 0:
            self.p[0] = sqrt(sum([pj**2 for pj in p[1:]]) + data.mass**2)
        self.h = float(h)
    ###########################################################################
    # Problem 4(c).
    ###########################################################################
    def w(self):
        """
        Return the Dirac spinor for this particle.
        """
        from math import sqrt
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
    ###########################################################################
    # Problem 4(d).
    ###########################################################################
    def wbar(self):
        """
        Return the bar Dirac spinor for this particle.
        """
        w = ~self.w()
        w[0], w[1], w[2], w[3] = w[2], w[3], w[0], w[1]
        return w

###############################################################################
# Problem 5.
###############################################################################
class Integrator:
    """
    This class integrates a two variable function.
    """
    ###########################################################################
    # Problem 4(a).
    ###########################################################################
    def __init__(self, f, xmin, xmax, ymin, ymax):
        """
        Initialize the integrator, given a function 'f', a minimum x
        'xmin', a maximum x 'xmax', a minumum y 'ymin', and a
        maxumimum y 'ymax.
        """
        self.f = f
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xdif = xmax - xmin
        self.ydif = ymax - ymin
    ###########################################################################
    # Problem 4(b).
    ###########################################################################
    def mc(self, n = 1000):
        """
        Perform MC integration for given number of sampling points 'n'.
        """
        import random
        t = 0
        for i in range(0, n):
            x = self.xmin + random.random()*self.xdif
            y = self.ymin + random.random()*self.ydif
            t += self.f(x, y)
        return t/float(n)*self.xdif*self.ydif

###############################################################################
# Problem 6.
###############################################################################
def circle(x, y):
    """
    Return 1 if 'x' and 'y' in a unit circle, 0 otherwise.
    """
    from math import sqrt
    f = sqrt(1 - x**2)
    return 0 if abs(y) > f else 1

###############################################################################
# Problem 7.
###############################################################################
class Annihilate:
    """
    This class defines the cross-section function needed to calculate
    the integrated cross-section of e+ e- -> mu+ mu-.
    """
    def __init__(self, p1, p2, p3, p4):
        """
        Initialize the 
        """
        from math import pi
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.dmu = DiracMatrices()
        self.dml = ~self.dmu
        # Calculate the cross-section prefactor ((hbar c)/(8 pi))^2 in
        # units m^2 GeV^2.
        self.xspre = (1.97326979e-16/(8*pi))**2
        # Calculate the matrix-element prefactor (-4 pi alpha).
        self.mepre = -4*pi/137.0
    def me(self):
        """
        Return the matrix element given the state of the internally
        represented particles.
        """
        p0 = self.p1.p + self.p2.p
        return self.mepre/p0**2*sum([
            (self.p3.wbar()*self.dmu[mu]*self.p4.w())*
            (self.p2.wbar()*self.dml[mu]*self.p1.w())
            for mu in range(0, 4)])
    def xs(self, phi, theta):
        """
        Return the cross-section in m^2 for a given phi and theta.
        """
        from math import sqrt, cos, sin
        ct = cos(theta)
        st = sin(theta)
        q = sqrt(self.p1.p[0]**2 - self.p3.data.mass**2)
        p = sqrt(sum([self.p1.p[mu]**2 for mu in range(1, 4)]))
        self.p3.p[0] = self.p1.p[0]
        self.p3.p[1] = q*st*cos(phi)
        self.p3.p[2] = q*st*sin(phi)
        self.p3.p[3] = q*ct
        self.p4.p = ~self.p3.p
        me = self.me()
        try: me2 = me.real**2 + me.imag**2
        except: me2 = me**2
        return self.xspre*me2*q/p*st/(self.p1.p[0] + self.p2.p[0])**2

###############################################################################
# Examples.
###############################################################################
if __name__== "__main__":
    def show(expr):
        """
        Print and evaluate an expression.
        """
        print("%s\n%10s:\n%s\n%s\n" % ("-"*20, expr, "-"*20, eval(expr)))

    # ParticleDatabase class.
    pdb = ParticleDatabase()
    show("pdb['e+']")
    show("pdb['e-']")

    # Check the anti-particles.
    pds = sorted([pd for pd in pdb.items() if type(pd[0]) == int])
    for pid, pd in pds:
        if pid > 0 and pid < 30 and not pd.anti: show("pd.name")
    
    # DiracMatrices class.
    dm = DiracMatrices()
    show("dm[0]")
    show("dm[1]")
    show("dm[2]")
    show("dm[3]")
    
    # Particle class.
    pp = FourVector(-1, 22.0, 150.0, 400)
    ph = 1
    p = Particle(pdb["e-"], pp, ph)
    show("p.w()")
    show("p.wbar()")
    p.h = -1
    show("p.w()")
    show("p.wbar()")
    
    # Diract momentum-space equation.
    m = Matrix([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])*abs(p.p)
    w = p.w()
    t = Matrix([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    for i in range(0, 4): t += dm[i]*(~p.p)[i]
    show("(t - m)*w")
    
    # Circle integration.
    i = Integrator(circle, -1, 1, -1, 1)
    show("i.mc()")

    # Helicity cross-sections.
    from math import pi
    p1 = FourVector(-1.0, 0.0, 0.0, 100)
    p2 = FourVector(-1.0, 0.0, 0.0, -100)
    p3 = FourVector(0.0, 0.0, 0.0, 0.0)
    p4 = FourVector(0.0, 0.0, 0.0, 0.0)
    pp1 = Particle(pdb["e-"], p1, 1) 
    pp2 = Particle(pdb["e+"], p2, 1) 
    pp3 = Particle(pdb["mu-"], p3, 1) 
    pp4 = Particle(pdb["mu+"], p4, 1) 
    a = Annihilate(pp1, pp2, pp3, pp4)
    i = Integrator(a.xs, 0.0, 2*pi, 0, pi)
    for h1 in [-1, 1]:
        pp1.h = h1
        for h2 in [-1, 1]:
            pp2.h = h2
            for h3 in [-1, 1]:
                pp3.h = h3
                for h4 in [-1, 1]:
                    pp4.h = h4
                    print("%2i %2i %2i %2i %8.1e" % (
                        h1, h2, h3, h4, i.mc(1000)/1e-31))
    
    # Cross-section integration.
    try: import pythia8; py = pythia8.Pythia("", False)
    except: py = None
    for p in [5.0, 10.0, 50.0, 100.0, 1000.0]:

        # Problem set cross-section.
        from math import pi
        p1 = FourVector(-1.0, 0.0, 0.0, p)
        p2 = FourVector(-1.0, 0.0, 0.0, -p)
        p3 = FourVector(0.0, 0.0, 0.0, 0.0)
        p4 = FourVector(0.0, 0.0, 0.0, 0.0)
        pp1 = Particle(pdb["e-"], p1, 1) 
        pp2 = Particle(pdb["e+"], p2, 1) 
        pp3 = Particle(pdb["mu-"], p3, 1) 
        pp4 = Particle(pdb["mu+"], p4, 1) 
        a = Annihilate(pp1, pp2, pp3, pp4)
        i = Integrator(a.xs, 0.0, 2*pi, 0, pi)
        xs0 = 0
        for h3 in [-1, 1]:
            pp3.h = h3
            for h4 in [-1, 1]:
                pp4.h = h4
                xs = []
                for h1 in [-1, 1]:
                    pp1.h = h1
                    for h2 in [-1, 1]:
                        pp2.h = h2
                        xs += [i.mc(1000)/1e-31]
                xs0 += sum(xs)/len(xs)
        show("xs0")
                
        # Pythia 8 cross-section.
        if py:
            py.readString("Print:quiet = on")
            py.readString("Beams:idA = 11")
            py.readString("Beams:idB = -11")
            py.readString("Beams:frameType = 3")
            py.readString("Beams:pzA = %r" % p)
            py.readString("Beams:pzB = -%r" % p)
            py.readString("PDF:lepton = off")
            py.readString("PartonLevel:all = off")
            py.readString("SigmaProcess:alphaEMorder = 0")
            py.readString("StandardModel:alphaEM0 = %r" % (1.0/137.0))
            py.readString("WeakSingleBoson:ffbar2ffbar(s:gm) = on")
            py.init()
            acc = 0
            for i in range(0, 10000):
                py.next()
                if py.process[5].idAbs() == 13: acc += 1
            xs1 = float(acc)/py.info.nAccepted()*py.info.sigmaGen()
        else: xs1 = 1.0
        show("p, xs1/xs0")
