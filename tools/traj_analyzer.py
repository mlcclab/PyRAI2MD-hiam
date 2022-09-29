## Molcas TSH Analyzer
## Jingbai Li Dec 30 2019
## version 0.1 Jingbai Li Jan 8  2020
## fix the redindex function Jan 9 2020 Jingbai Li
## add axis swap and reflection to hungarian algorithm Jan 9 2020 Jinbai Li
## analysis part is parallelized Jan 9 2020 Jingbai Li
## read in part is parallelized Jan 10 2020 Jingbai Li
## minor fix in read_raw_data and add cutoff in read Jan 13 2020 Jingbai Li
## add diagnosis function Jan 14 2020 Jingbai Li
## add sorted prod list and minor fix in count_data_lines and read_raw_data Jan 16 2020 Jingbai Li
## add function to print the coordinates of selected products; add initial state to label;add traj name to data set Feb 17 2020 Jingbai Li
## fix bug in reading control file Feb 19 2020 Jingbai Li
## support NewtonX/BAGEL output Aug 17 2020 Jingbai Li
## add direction to dehidral angle Oct 30 2020 Jingbai Li
## add function to save population to txt Dec 8 2020 Jingbai Li
## add function to save kinetic energy to txt Feb 18 2021 Jingbai Li
## add D2 paramter to measutre dihedral involving dummy center Mar 17 Jingbai Li
## add prune function to canonicalize trajectory data Mar 19 2021 Jingbai Li
## add repair function to average incomplete trajectories Mar 19 Jingbai Li
## add atom selection in RMSD Mar 29 Jingbai Li
## add option to not save data in json
## support old version <19 of Molcas Oct 21 2021 Jingbai Li
## support SHARC output Nov 16 2021 Jingbai Li
## fix read PyRAI2MD log Dec 13 2021 Jingbai Li
## support fromage_dynamics Feb 28 2022 Jingbai Li

import sys,os,json,multiprocessing
import numpy as np
from numpy import linalg as la
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
from optparse import OptionParser

class Element:

    def __init__(self,name):

        Periodic_Table = {
             "HYDROGEN":"1","H":"1","1":"1",
             "HELIUM":"2","He":"2","2":"2","HE":"2",
             "LITHIUM":"3","Li":"3","3":"3","LI":"3",
             "BERYLLIUM":"4","Be":"4","4":"4","BE":"4",
             "BORON":"5","B":"5","5":"5",
             "CARBON":"6","C":"6","6":"6",
             "NITROGEN":"7","N":"7","7":"7",
             "OXYGEN":"8","O":"8","8":"8",
             "FLUORINE":"9","F":"9","9":"9",
             "NEON":"10","Ne":"10","10":"10","NE":"10",
             "SODIUM":"11","Na":"11","11":"11","NA":"11",
             "MAGNESIUM":"12","Mg":"12","12":"12","MG":"12",
             "ALUMINUM":"13","Al":"13","13":"13","AL":"12",
             "SILICON":"14","Si":"14","14":"14","SI":"14",
             "PHOSPHORUS":"15","P":"15","15":"15",
             "SULFUR":"16","S":"16","16":"16",
             "CHLORINE":"17","Cl":"17","17":"17","CL":"17",
             "ARGON":"18","Ar":"18","18":"18","AG":"18",
             "POTASSIUM":"19","K":"19","19":"19",
             "CALCIUM":"20","Ca":"20","20":"20","CA":"20",
             "SCANDIUM":"21","Sc":"21","21":"21","SC":"21",
             "TITANIUM":"22","Ti":"22","22":"22","TI":"22",
             "VANADIUM":"23","V":"23","23":"23",
             "CHROMIUM":"24","Cr":"24","24":"24","CR":"24",
             "MANGANESE":"25","Mn":"25","25":"25","MN":"25",
             "IRON":"26","Fe":"26","26":"26","FE":"26",
             "COBALT":"27","Co":"27","27":"27","CO":"27",
             "NICKEL":"28","Ni":"28","28":"28","NI":"28",
             "COPPER":"29","Cu":"29","29":"29","CU":"29",
             "ZINC":"30","Zn":"30","30":"30","ZN":"30",
             "GALLIUM":"31","Ga":"31","31":"31","GA":"31",
             "GERMANIUM":"32","Ge":"32","32":"32","GE":"32",
             "ARSENIC":"33","As":"33","33":"33","AS":"33",
             "SELENIUM":"34","Se":"34","34":"34","SE":"34",
             "BROMINE":"35","Br":"35","35":"35","BR":"35",
             "KRYPTON":"36","Kr":"36","36":"36","KR":"36",
             "RUBIDIUM":"37","Rb":"37","37":"37","RB":"37",
             "STRONTIUM":"38","Sr":"38","38":"38","SR":"38",
             "YTTRIUM":"39","Y":"39","39":"39",
             "ZIRCONIUM":"40","Zr":"40","40":"40","ZR":"40",
             "NIOBIUM":"41","Nb":"41","41":"41","NB":"41",
             "MOLYBDENUM":"42","Mo":"42","42":"42","MO":"42",
             "TECHNETIUM":"43","Tc":"43","43":"43","TC":"43",
             "RUTHENIUM":"44","Ru":"44","44":"44","RU":"44",
             "RHODIUM":"45","Rh":"45","45":"45","RH":"45",
             "PALLADIUM":"46","Pd":"46","46":"46","PD":"46",
             "SILVER":"47","Ag":"47","47":"47","AG":"47",
             "CADMIUM":"48","Cd":"48","48":"48","CD":"48",
             "INDIUM":"49","In":"49","49":"49","IN":"49",
             "TIN":"50","Sn":"50","50":"50","SN":"50",
             "ANTIMONY":"51","Sb":"51","51":"51","SB":"51",
             "TELLURIUM":"52","Te":"52","52":"52","TE":"52",
             "IODINE":"53","I":"53","53":"53",
             "XENON":"54","Xe":"54","54":"54","XE":"54",
             "CESIUM":"55","Cs":"55","55":"55","CS":"55",
             "BARIUM":"56","Ba":"56","56":"56","BA":"56",
             "LANTHANUM":"57","La":"57","57":"57","LA":"57",
             "CERIUM":"58","Ce":"58","58":"58","CE":"58", 
             "PRASEODYMIUM":"59","Pr":"59","59":"59","PR":"59",
             "NEODYMIUM":"60","Nd":"60","60":"60","ND":"60", 
             "PROMETHIUM":"61","Pm":"61","61":"61","PM":"61", 
             "SAMARIUM":"62","Sm":"62","62":"62","SM":"62",
             "EUROPIUM":"63","Eu":"63","63":"63","EU":"63", 
             "GADOLINIUM":"64","Gd":"64","64":"64","GD":"64", 
             "TERBIUM":"65","Tb":"65","65":"65","TB":"65",
             "DYSPROSIUM":"66","Dy":"66","66":"66","DY":"66", 
             "HOLMIUM":"67","Ho":"67","67":"67","HO":"67", 
             "ERBIUM":"68","Er":"68","68":"68","ER":"68", 
             "THULIUM":"69","TM":"69","69":"69","TM":"69", 
             "YTTERBIUM":"70","Yb":"70","70":"70","YB":"70", 
             "LUTETIUM":"71","Lu":"71","71":"71","LU":"71",
             "HAFNIUM":"72","Hf":"72","72":"72","HF":"72",
             "TANTALUM":"73","Ta":"73","73":"73","TA":"73",
             "TUNGSTEN":"74","W":"74","74":"74",
             "RHENIUM":"75","Re":"75","75":"75","RE":"75",
             "OSMIUM":"76","Os":"76","76":"76","OS":"76",
             "IRIDIUM":"77","Ir":"77","77":"77","IR":"77",
             "PLATINUM":"78","Pt":"78","78":"78","PT":"78",
             "GOLD":"79","Au":"79","79":"79","AU":"79",
             "MERCURY":"80","Hg":"80","80":"80","HG":"80",
             "THALLIUM":"81","Tl":"81","81":"81","TL":"81",
             "LEAD":"82","Pb":"82","82":"82","PB":"82",
             "BISMUTH":"83","Bi":"83","83":"83","BI":"83",
             "POLONIUM":"84","Po":"84","84":"84","PO":"84",
             "ASTATINE":"85","At":"85","85":"85","AT":"85",
             "RADON":"86","Rn":"86","86":"86","RN":"86"}

        FullName=["HYDROGEN", "HELIUM", "LITHIUM", "BERYLLIUM", "BORON", "CARBON", "NITROGEN", "OXYGEN", "FLUORINE", "NEON", 
              "SODIUM", "MAGNESIUM", "ALUMINUM", "SILICON", "PHOSPHORUS", "SULFUR", "CHLORINE", "ARGON", "POTASSIUM", "CALCIUM", 
              "SCANDIUM", "TITANIUM", "VANADIUM", "CHROMIUM", "MANGANESE", "IRON", "COBALT", "NICKEL", "COPPER", "ZINC", 
              "GALLIUM", "GERMANIUM", "ARSENIC", "SELENIUM", "BROMINE", "KRYPTON", "RUBIDIUM", "STRONTIUM", "YTTRIUM", "ZIRCONIUM", 
              "NIOBIUM", "MOLYBDENUM", "TECHNETIUM", "RUTHENIUM", "RHODIUM", "PALLADIUM", "SILVER", "CADMIUM", "INDIUM", "TIN", 
              "ANTIMONY", "TELLURIUM", "IODINE", "XENON", "CESIUM", "BARIUM", "LANTHANUM", "CERIUM", "PRASEODYMIUM", "NEODYMIUM", 
              "PROMETHIUM", "SAMARIUM", "EUROPIUM", "GADOLINIUM", "TERBIUM", "DYSPROSIUM", "HOLMIUM", "ERBIUM", "THULIUM", "YTTERBIUM", 
              "LUTETIUM", "HAFNIUM", "TANTALUM", "TUNGSTEN", "RHENIUM", "OSMIUM", "IRIDIUM", "PLATINUM", "GOLD", "MERCURY", 
              "THALLIUM", "LEAD", "BISMUTH", "POLONIUM", "ASTATINE", "RADON"]

        Symbol=[ "H","He","Li","Be","B","C","N","O","F","Ne",
                "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
                "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
                "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
                "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","TM","Yb",
                "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
                "Tl","Pb","Bi","Po","At","Rn"]

        Mass=[1.008,4.003,6.941,9.012,10.811,12.011,14.007,15.999,18.998,20.180,
              22.990,24.305,26.982,28.086,30.974,32.065,35.453,39.948,39.098,40.078,
              44.956,47.867,50.942,51.996,54.938,55.845,58.933,58.693,63.546,65.390,
              69.723,72.640,74.922,78.960,79.904,83.800,85.468,87.620,88.906,91.224,
              92.906,95.940,98.000,101.070,102.906,106.420,107.868,112.411,114.818,118.710,
              121.760,127.600,126.905,131.293,132.906,137.327,138.906,140.116,140.908,144.240,
              145.000,150.360,151.964,157.250,158.925,162.500,164.930,167.259,168.934,173.040,
              174.967,178.490,180.948,183.840,186.207,190.230,192.217,195.078,196.967,200.590,
              204.383,207.200,208.980,209.000,210.000,222.000]

        # Van der Waals Radius, missing data replaced by 2.00
        Radii=[1.20,1.40,1.82,1.53,1.92,1.70,1.55,1.52,1.47,1.54,
               2.27,1.73,1.84,2.10,1.80,1.80,1.75,1.88,2.75,2.31,
               2.11,2.00,2.00,2.00,2.00,2.00,2.00,1.63,1.40,1.39,
               1.87,2.11,1.85,1.90,1.85,2.02,3.03,2.49,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,1.63,1.72,1.58,1.93,2.17,
               2.00,2.06,1.98,2.16,3.43,2.68,2.00,2.00,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,2.00,2.00,1.75,1.66,1.55,
               1.96,2.02,2.07,1.97,2.02,2.20]

        self.__name = int(Periodic_Table[name])
        self.__FullName = FullName[self.__name-1]
        self.__Symbol = Symbol[self.__name-1]
        self.__Mass = Mass[self.__name-1]
        self.__Radii = Radii[self.__name-1]

    def getFullName(self):
        return self.__FullName
    def getSymbol(self):
        return self.__Symbol
    def getUpperSymbol(self):
        return self.__Symbol.upper()
    def getMass(self):
        return self.__Mass
    def getNuc(self):
        return self.__name
    def getNelectron(self):
        return self.__name
    def getRadii(self):
        return self.__Radii

def BND(xyz,var):
    ## This function calculate distance
    ## a<->b

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b=var[0:2]

    v1=xyz[a-1]
    v2=xyz[b-1]
    r=la.norm(v1-v2)
    return r

def AGL(xyz,var):
    ## This function calculate angle
    ## a<-b->c

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b,c=var[0:3]

    r1=np.array(xyz[a-1])
    r2=np.array(xyz[b-1])
    r3=np.array(xyz[c-1])
    v1=r1-r2
    v2=r3-r2
    v1=v1/la.norm(v1)
    v2=v2/la.norm(v2)
    cosa=np.dot(v1,v2)
    alpha=np.arccos(cosa)*57.2958
    return alpha

def DHD(xyz,var):
    ## This function calculate dihedral angle
    ##   n1    n2
    ##    |    |
    ## a<-b-><-c->d

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b,c,d=var[0:4]
    r1=np.array(xyz[a-1])
    r2=np.array(xyz[b-1])
    r3=np.array(xyz[c-1])
    r4=np.array(xyz[d-1])
    v1=r1-r2
    v2=r3-r2
    v3=r2-r3
    v4=r4-r3
    n1=np.cross(v1,v2)
    n2=np.cross(v3,v4)
    n1=n1/la.norm(n1)
    n2=n2/la.norm(n2)
    cosb=np.dot(n1,n2)
    beta=np.arccos(cosb)*57.2958
    axis=np.cross(n1,n2)
    pick=np.argmax(np.abs(axis))
    sign=np.sign(axis[pick]/v2[pick])  # find the projection with largest magnitude (non-zero), then just compare it to avoid 0/0
    if sign == -1:
        beta=360-beta

    return beta

def DHD2(xyz,var):
    ## This function calculate dihedral angle involving dummpy center
    ##   n1    n2
    ##    |    |
    ## a,b<-c-><-d->e,f

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b,c,d,e,f=var[0:6]
    r1=np.array(xyz[a-1])
    r2=np.array(xyz[b-1])
    r3=np.array(xyz[c-1])
    r4=np.array(xyz[d-1])
    r5=np.array(xyz[e-1])
    r6=np.array(xyz[f-1])
    v1=(r1+r2)/2-r3
    v2=r4-r3
    v3=r3-r4
    v4=(r5+r6)/2-r4
    n1=np.cross(v1,v2)
    n2=np.cross(v3,v4)
    n1=n1/la.norm(n1)
    n2=n2/la.norm(n2)
    cosb=np.dot(n1,n2)
    beta=np.arccos(cosb)*57.2958
    axis=np.cross(n1,n2)
    pick=np.argmax(np.abs(axis))
    sign=np.sign(axis[pick]/v2[pick])  # find the projection with largest magnitude (non-zero), then just compare it to avoid 0/0
    if sign == -1:
        beta=360-beta

    return beta

def OOP(xyz,var):
    ## This function calculate out-of-plane angle
    ##    n  d
    ##    |  |   
    ## a<-b->c 

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b,c,d=var[0:4]
    r1=np.array(xyz[a-1])
    r2=np.array(xyz[b-1])
    r3=np.array(xyz[c-1])
    r4=np.array(xyz[d-1])
    v1=r1-r2
    v2=r3-r2
    v3=r4-r3
    v3=v3/la.norm(v3)
    n=np.cross(v1,v2)
    n=n/la.norm(n)
    cosb=np.dot(n,v3)
    gamma=np.arccos(cosb)*57.2958

    return gamma

def PPA(xyz,var):
    ## This function calculate plane-plane angle
    ##   n1       n2
    ##    |       |
    ## a<-b->c d<-e->f

    var=[int(x) for x in var]
    xyz=np.array([[float(y) for y in x.split()[1:]] for x in xyz])
    a,b,c,d,e,f=var[0:6]
    r1=np.array(xyz[a-1])
    r2=np.array(xyz[b-1])
    r3=np.array(xyz[c-1])
    r4=np.array(xyz[d-1])
    r5=np.array(xyz[e-1])
    r6=np.array(xyz[f-1])
    v1=r1-r2
    v2=r3-r2
    v3=r4-r5
    v4=r6-r5
    n1=np.cross(v1,v2)
    n2=np.cross(v3,v4)
    n1=n1/la.norm(n1)
    n2=n2/la.norm(n2)
    cosb=np.dot(n1,n2)
    delta=np.arccos(cosb)*57.2958
    return delta

def G(coord,par_sym):
    ## This function calculate symmetry function in Behler, J, Int. J. Quantum Chem., 2-15, 115 1032-1050
    ## This function return a list of values for each atom
    ## coord is a numpy array of floating numbers 
    ## par_sym has default values in RMSD

    cut=par_sym['cut'] ## cutoff function version 1 or 2
    ver=par_sym['ver'] ## symmetry function 1-4
    rc=par_sym['rc']   ## cutoff radii, 0 is the maximum * 1.1
    n=par_sym['n']     ## Gaussian exponent
    rs=par_sym['rs']   ## Gaussian center
    z=par_sym['z']     ## anglular exponent
    l=par_sym['l']     ## cosine exponent, only 1 or -1

    #print('\nSymmetry function: %d' % (ver))
    #print('Cutoff function: %d' % (cut))
    #print('Cutoff radii:%6.2f Shift:%6.2f' % (rc,rs))
    #print('Eta:%6.2f Zeta:%6.2f Lambda:%6.2f' % (n,z,l))

    ## prepare distance matrix
    dist=np.array([[0.0 for x in coord] for x in coord])
    for n,i in enumerate(coord):
        for m,j in enumerate(coord):
            if n != m:        # update dist if n != m
                r=la.norm(i-j)
                dist[n][m]=r

    ## prepare cutoff function matrix
    if rc == 0:
        rc=1.1*np.amax(dist)
    fc=np.array([[1.0 for x in coord] for x in coord])
    for n,i in enumerate(dist):
        for m,j in enumerate(dist):
            r/=rc
            if r < 1:
                fc[n][m]=r    # update fc if r < rc

    ## prepare angle matrix if needed, i is the center atom!
    if ver > 2:
        angl=np.array([[[0.0 for x in coord] for x in coord] for x in coord])    
        for n,i in enumerate(coord):
            for m,j in enumerate(coord):
                for l,k in enumerate(coord):
                    if n !=m and m !=l and l != n:
                        v1=j-i
                        v2=j-l
                        v1=v1/la.norm(v1)
                        v2=v2/la.norm(v2)
                        cosa=np.dot(v1,v2)
                        alpha=np.arccos(cosa)*57.2958
                        angl[n][m][l]=alpha
    if   cut == 1:
        fc=0.5*np.cos(np.pi*fc)+1
    elif cut == 2:
        fc=np.tanh(1-fc)**3
    else:
        print('\n!!! Cannot recognize cutoff function !!!\n')
        exit()

    if   ver == 1:
        g=np.sum(fc,axis=1)
    elif ver == 2:
        w=np.exp((-1)*n*(dist-rs)**2)
       	g=np.sum(w*fc,axis=1)
    elif ver == 3:
        g=np.array([0.0 for x in coord])
        for i in range(len(coord)):
            for j in range(len(coord)):
                for k in range(len(coord)):
                    a=(1+l*np.cos(angl[i][j][k]))**z
                    w=np.exp((-1)*n*(dist[i][j]**2+dist[i][k]**2+dist[j][k]**2))
                    f=fc[i][j]*fc[i][k]*fc[j][k]
                    g[i]+=2**(1-z)*a*w*f
    elif ver == 4:
       	g=np.array([0.0 for x in coord])
       	for i in range(len(coord)):
            for j in range(len(coord)):
                for k in range(len(coord)):
       	       	    a=(1+l*np.cos(angl[i][j][k]))**z
                    w=np.exp((-1)*n*(dist[i][j]**2+dist[i][k]**2))
       	       	    f=fc[i][j]*fc[i][k]
       	       	    g[i]+=2**(1-z)*a*w*f
    else:
        print('\n!!! Cannot recognize symmetry function !!!\n')
        exit()
    return g

def RMSD(xyz,ref,var):
    ## This function calculate RMSD between product and reference
    ## This function call kabsch to reduce RMSD between product and reference
    ## This function call hungarian to align product and reference

    ## general variables for all functions
    excl=[]      ## exclude elements
    incl=[]      ## only inlcude elements
    pck=[]       ## pick this atoms
    align='NO'   ## do not align product and reference
    coord='CART' ## use cartesian coordinates
    rmsd='NONE'  ## rmsd have not been calculated

    ## symmetry function default variables
    par_sym={
    'cut':1,     ## cutoff function version 1 or 2
    'ver':1,     ## symmetry function 1-4
    'rc': 6,     ## cutoff radii, 0 is the maximum * 1.1
    'n':  1.2,   ## Gaussian exponent
    'rs': 0,     ## Gaussian center
    'z':  1,     ## anglular exponent
    'l':  1      ## cosine factor, only 1 or -1
    }

    for i in var:
        i=i.upper()
        if   'NO=' in i:    
            e=i.split('=')[1]
            e=Element(e).getSymbol()
            excl.append(e)
        elif 'ON=' in i:
            e=i.split('=')[1]
            e=Element(e).getSymbol()
            incl.append(e)
        elif 'PICK=' in i:
            pck=[int(x) for x in i.split('=')[1].split(',')]
        elif 'ALIGN=' in i:
            i=i.split('=')[1]
            if i == 'HUNG' or i == 'NO':  ## align must be either hung or no
                align=i
        elif 'COORD=' in i:
            i=i.split('=')[1]
            if i == 'CART' or i == 'SYM': ## coord must be either cart or sym
                coord=i
        elif 'CUT=' in i:
            i=int(i.split('=')[1])
            if i in [1,2]:                ## cut must be within 1-2
                par_sym['cut']=i
        elif 'VER=' in i:
            i=int(i.split('=')[1])
            if i in [1,2,3,4]:            ## ver must be within 1-4
                par_sym['ver']=i
        elif 'RC=' in i:
            par_sym['rc']=float(i.split('=')[1])
        elif 'ETA=' in i:
            par_sym['n']=float(i.split('=')[1])
        elif 'RS=' in i:
            par_sym['rs']=float(i.split('=')[1])
        elif 'ZETA=' in i:
            par_sym['z']=float(i.split('=')[1])
        elif 'LAMBDA=' in i:
            l=int(i.split('=')[1])
            if l >= 0:                    ## l only takes +1 or -1
                l=1
            else:
                l=-1
            par_sym['l']=l

    ## prepare atom list and coordinates 
    E=[]                     ## element list
    for i in xyz:
        e,x,y,z=i.split()
        e=Element(e).getSymbol()
       	if e not in E:
       	    E.append(e)

    if len(excl)>0:
        E=[x for x in E if x not in excl]
    if len(incl)>0:
        E=[x for x in E if x in incl]

    S=[x+1 for x in range(len(xyz))] ## atom index list
    if len(pck)>0:
        S=pck

    P=[]                     ## product coordinates
    Patoms=[]
    for n,i in enumerate(xyz):
        e,x,y,z=i.split()
        x,y,z=float(x),float(y),float(z)
        e=Element(e).getSymbol()
        if e in E and n+1 in S:
            P.append([x,y,z])
            Patoms.append(e)
    P=np.array(P)

    Q=[]                     ## reference coordinates
    Qatoms=[]
    for n,i in enumerate(ref):
        e,x,y,z=i.split()
        x,y,z=float(x),float(y),float(z)
        e=Element(e).getSymbol()
       	if e in E and n+1 in S:
            Q.append([x,y,z])
            Qatoms.append(e)
    Q=np.array(Q)

    P-=P.mean(axis=0)        ## translate to the centroid
    Q-=Q.mean(axis=0)        ## translate to the centroid


    if align == 'HUNG':      ## align coordinates
        swap = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]])

        reflection = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1]])

        order=[]
        rmsd=[]
        for s in swap:
            for r in reflection:
                Tatoms=[x for x in Qatoms]
                T =np.array([x for x in Q])
                T =T[:, s]
                T =np.dot(T, np.diag(r))
                T-=T.mean(axis=0)
                Ip=inertia(Patoms,P)
                It=inertia(Tatoms,T)
                U1=rotate(Ip,It)
                U2=rotate(Ip,-It)
                T1=np.dot(T,U1)
                T2=np.dot(T,U2)
                order1 = hungarian(Patoms, Tatoms, P, T1)
                order2 = hungarian(Patoms, Tatoms, P, T2)
                rmsd1=kabsch(P,T[order1])
                rmsd2=kabsch(P,T[order2])
                order+=[order1,order2]
                rmsd+=[rmsd1,rmsd2] 
        pick=np.argmin(rmsd)
       	order=order[pick]
        rmsd=rmsd[pick]
        Q=Q[order]
    if coord == 'SYM':       ## use symmetry function 
        g_prd=G(P,par_sym)
        g_ref=G(Q,par_sym)
        rmsd=np.sqrt(np.sum((g_prd-g_ref)**2)/len(g_prd))

    if rmsd == 'NONE':
        rmsd=kabsch(P,Q)

    return rmsd

def kabsch(P,Q):
    ## This function use Kabsch algorithm to reduce RMSD by rotation

    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:                    # ensure right-hand system
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P = np.dot(P, U)
    diff = P-Q
    N = len(P)
    return np.sqrt((diff * diff).sum() / N)

def inertia(atoms,xyz):
    ## This function calculate principal axis

    xyz=np.array([i for i in xyz])   # copy the array to avoid changing it
    mass=[]
    for i in atoms:
        m=Element(i).getMass()
        mass.append(m)
    mass=np.array(mass)
    xyz-=np.average(xyz,weights=mass,axis=0)
    xx=0.0
    yy=0.0
    zz=0.0
    xy=0.0
    xz=0.0
    yz=0.0
    for n,i in enumerate(xyz):
        xx+= mass[n]*(i[1]**2+i[2]**2)
       	yy+= mass[n]*(i[0]**2+i[2]**2)
       	zz+= mass[n]*(i[0]**2+i[1]**2)
       	xy+=-mass[n]*i[0]*i[1]
       	xz+=-mass[n]*i[0]*i[2]
       	yz+=-mass[n]*i[1]*i[2]

    I=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    eigval,eigvec=np.linalg.eig(I)

    return eigvec[np.argmax(eigval)]

def rotate(p,q):
    ## This function calculate the matrix rotate q onto p
 
    if (p == q).all():
        return np.eye(3)
    elif (p == -q).all():
        # return a rotation of pi around the y-axis
        return np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
    else:
        v = np.cross(p, q)
        s = np.linalg.norm(v)
        c = np.vdot(p, q)
        vx = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        return np.eye(3) + vx + np.dot(vx,vx)*((1.-c)/(s*s))

def hungarian(Patoms,Qatoms,P,Q):
    ## This function use hungarian algorithm to align P onto Q
    ## This function call linear_sum_assignment from scipy to solve hungarian problem
    ## This function call inertia to find principal axis
    ## This function call rotate to rotate P onto aligned Q

    unique_atoms=np.unique(Patoms)

    reorder=np.zeros(len(Qatoms),dtype=int)
    for atom in unique_atoms:
        Pidx=[]
        Qidx=[]

        for n,p in enumerate(Patoms):
            if p == atom:
                Pidx.append(n)
        for m,q in enumerate(Qatoms):
            if q == atom:
                Qidx.append(m)

        Pidx=np.array(Pidx)
        Qidx=np.array(Qidx)
        A=P[Pidx]
        B=Q[Qidx]
        AB=np.array([[la.norm(a-b) for b in B] for a in A])
        Aidx,Bidx = linear_sum_assignment(AB)
        reorder[Pidx] = Qidx[Bidx]
    return reorder

def getindex(index):
    ## This function read single, range, separete range index and convert them to a list
    index_list=[]
    for i in index:
        if '-' in i:
            a,b=i.split('-')
            a,b=int(a),int(b)
            index_list+=range(a,b+1)
        else:
            index_list.append(int(i))    

    index_list=sorted(list(set(index_list))) # remove duplicates and sort from low to high
    return index_list

def redindex(index):
    ## This function compress a list of index into range
    index=sorted(list(set(index)))
    groups=[]
    subrange=[]
    for i in index:
        subrange.append(int(i))
        if len(subrange) > 1:
            d=subrange[-1]-subrange[-2] # check continuity
            if d > 1:
                groups.append(subrange[0:-1])
                subrange=[subrange[-1]]
        if i == index[-1]:
            groups.append(subrange)

    index_range=''
    for j in groups:
        if   len(j) == 1:
            index_range+='%s '    % (j[0])
        elif len(j) == 2:
            index_range+='%s %s ' % (j[0],j[1])
        else:
            index_range+='%s-%s ' % (j[0],j[-1])
    return index_range

def format(n,xyz):
    ## This function convert coordinates list to string
    output='%d\n' % (n)
    for i in xyz:
        output+='%s\n' % (i)
    return output

def format2(str):
    ## This function convert a one-line string to multiple lines:
    str_new=''
    for n,i in enumerate(str.split()):
        str_new+='%10s ' % (i) 
        if (n+1)%10 == 0:
            str_new+='\n'
        else:
            if i == str.split()[-1]:
                str_new+='\n'
    return str_new

def Refread(natom,ref):
    ## This function read the reference structure from a file: ref    
    ref_coord=[]
    with open(ref,'r') as refxyz:
        coord=refxyz.read().splitlines()
    n=0
    m=0
    for line in coord:
        n+=1
        if n % (natom+2) == 0: # at the last line of each coordinates
            m+=1
            ref_coord.append(coord[n-natom:n])
    print('\nRead reference structures: %5d in %s\n' % (m,ref))
    return ref_coord

def Trkread(track):
    ## This function read the tracking parameters from string or a file track
    par_track=track

    P=['B','A','D','D2','O','P','RMSD']
    parameters=[]
    par_group=[]

    for	n,p in enumerate(par_track):
       	if p in	P:
       	    if len(par_group) >	0:
       	        parameters.append(par_group)
       	    par_group=[]
       	    par_group.append(p)
       	else:
       	    par_group.append(p)
       	if n == len(par_track)-1:
            parameters.append(par_group)

    choose=-1 # choose one of the RMSD as standard 
    pos=-1     # position of RMSD in parameters
    for n,i in enumerate(parameters):
        n+=1
        i=[j.upper() for j in i]
        if i[0] == 'RMSD':
            pos=n
            if 'X' in i:
                choose=n

    if pos > -1 and choose == -1:
        choose=pos    

    return parameters,choose

def Pararead(parameters):
    ## This function read track parameters and return the name and comment of parameters
    func={'B':'Bond','A':'Angle','D':'Diheral','O':'Out-of-Plane','P':'Plane-Plane','RMSD':'RMSD'}
    name=[]
    cmmt=[]
    for i in parameters:
        i=[x.upper() for x in i]
        name.append(i[0])
        if i[0] != 'RMSD':
            cmmt.append(','.join(i[1:]))
        else:
            v1='D'
            v2=' CART'
            if 'ALIGN=HUNG' in i:
                v1='A'
            if 'COORD=SYM'  in i:
                v2='  SYM'
            cmmt.append(v1+v2)
    return name,cmmt

def evaluate(prd,ref,par,diff):
    ## This function evaluate the value of the parameter given a coordinate
    func=par[0]
    var=par[1:] 
    ref_val=0
    prd_val=0
    dev_val=0
    if   func == 'B':
        method=BND
    elif func == 'A':
        method=AGL
    elif func == 'D':
        method=DHD
    elif func == 'D2':
        method=DHD2
    elif func == 'O':
        method=OOP
    elif func == 'P':
        method=PPA
    elif func == 'RMSD':
        method=RMSD
    else:
        print('Method is not found: %s' % (func))
        exit()

    if func == 'RMSD':
        prd_val=method(prd,ref,var)
        dev_val=prd_val
    else:
        prd_val=method(prd,var)
        if diff != 0:
            ref_val=method(ref,var)
            dev_val=ref_val-prd_val
            dev_val*=(-1)**(dev_val<0) # absolute value

    return prd_val,ref_val,dev_val

def HOPdiag(files,prog):
    ## This function run diagnosis for calculation results
    ## This function call count_data_lines
    ## This function print lines number of all required data for HOPread
    ## This function will classify files if mstep != 0

    diag_func={
    'molcas'  : count_data_lines_molcas,
    'newtonx' : count_data_lines_newtonx,
    'sharc'   : count_data_lines_sharc,
    'fromage' : count_data_lines_fromage,
    }
    procs=cpus
    input_val=[]
    for n,f in enumerate(files):
        input_val.append([n,f])

    result=[[] for x in range(len(input_val))]

    if (len(input_val)) < procs:
        procs=len(input_val)
    sys.stdout.write('CPU: %3d Checking data: \r' % (procs))
    pool=multiprocessing.Pool(processes=procs)
    for ntraj, val in enumerate(pool.imap_unordered(diag_func[prog],input_val)):
        ntraj+=1
        p,name,nlog,nenergy,nxyz=val
        result[p]=[name,nlog,nenergy,nxyz]
        sys.stdout.write('CPU: %3d Checking data: %6.2f%% %d/%d\r' % (procs,ntraj*100/(len(input_val)),ntraj,len(input_val)))

    select=[]

    print('\nDiagnosis results\n%-20s%12s%12s%12s\n' % ('Name','.log','.md.energy','.md.xyz'))
    for i in result:
        print('%-20s%12d%12d%12d' % (i[0],i[1],i[2],i[3]))
        if i[1] >= mstep and i[2] >= mstep and i[3] >= mstep and mstep != 0:
            select.append(i)

    if mstep != 0:
        print('\nRequired steps: %5d\nData satisfy the requirement\n' % (mstep))
        for i in select:
            print('%s' % (i[0]))

def count_data_lines_molcas(files):
    ## This function count lines for molcas data from provided folders 
    ## This function is for parallelization
    ntraj,f=files
    t=f.split('/')[-1]
    if os.path.exists('%s/%s.log' % (f,t)) == True:
        with open('%s/%s.log' % (f,t),'r') as logfile:
            log=logfile.read().splitlines()
        n=0
        for line in log:
            if 'Gnuplot' in line:
                n+=1
        nlog=n
    elif os.path.exists('%s/%s.out' % (f,t)) == True:
        with open('%s/%s.out' % (f,t),'r') as logfile:
            log=logfile.read().splitlines()
        n=0
        for line in log:
            if 'Gnuplot' in line:
                n+=1
        nlog=n
    else:
        nlog=0

    if os.path.exists('%s/%s.md.energies' % (f,t)) == True:
        with open('%s/%s.md.energies' % (f,t),'r') as engfile:
            eng=engfile.read().splitlines()
        nenergy=len(eng)-1
    else:
        nenergy=0

    if os.path.exists('%s/%s.md.xyz' % (f,t)) == True:
        with open('%s/%s.md.xyz' % (f,t),'r') as xyzfile:
            xyz=xyzfile.read().splitlines()
        natom=int(xyz[0])
        nxyz=len(xyz)/(natom+2)
    else:
        nxyz=0

    return ntraj,f,nlog,nenergy,nxyz

def count_data_lines_newtonx(files):
    ## This function count lines for newtonx data from provided folders
    ## This function is for parallelization
    ntraj,f=files
    if os.path.exists('%s/RESULTS/dyn.out' % (f)) == True:
        with open('%s/RESULTS/dyn.out' % (f),'r') as logfile:
            log=logfile.read().splitlines()
        n=0
        for line in log:
            if 'STEP' in line:
                n+=1
        nlog=n-1
        nxyz=n-1
    else:
        nlog=0
        nxyz=0

    if os.path.exists('%s/RESULTS/en.dat' % (f)) == True:
        with open('%s/RESULTS/en.dat' % (f),'r') as engfile:
            eng=engfile.read().splitlines()
        nenergy=len(eng)
    else:
        nenergy=0

    return ntraj,f,nlog,nenergy,nxyz

def count_data_lines_sharc(files):
    ## This function count lines for sharc data from provided folders
    ## This function is for parallelization
    ntraj,f=files
    if os.path.exists('%s/output.dat' % (f)) == True:
        with open('%s/output.dat' % (f),'r') as logfile:
            log=logfile.read().splitlines()
        n=0
        for line in log:
            if '! 0 Step' in line:
                n+=1
        nlog = n - 1
        nenergy = n - 1
        nxyz = n - 1
    else:
        nlog = 0
        nenergy = 0
        nxyz = 0

    return ntraj,f,nlog,nenergy,nxyz

def count_data_lines_fromage(files):
    ## This function count lines for fromage data from provided folders
    ## This function is for parallelization
    ntraj,f=files
    if os.path.exists('%s/output.chk' % (f)) == True:
        with open('%s/output.chk' % (f),'r') as logfile:
            log=logfile.read().splitlines()
        n = len(log) - 3
        nlog = n
        nenergy = n
        nxyz = n
    else:
        nlog = 0
        nenergy = 0
        nxyz = 0

    return ntraj,f,nlog,nenergy,nxyz

def HOPread(title,files,prog):
    ## This function read data from calculation folders
    ## This function call read_raw_data
    ## This function return a dictionary for lists of natom, nstate, nstep, dtime, kin, pot, pop, label, coord, hop, prod

    read_func={
    'molcas'  : read_raw_data_molcas,
    'newtonx' : read_raw_data_newtonx,
    'sharc'   : read_raw_data_sharc,
    'fromage' : read_raw_data_fromage,
    }
    ## initialize variables
    procs=cpus
    natom=0
    nstate=0
    ntraj=0
    nstep=0
    dtime=0

    ## A parallelized loop goes over all trajetories to find trj_kin,trj_pot,trj_pop,trj_coord,trj_hop in each trajectory and appends to
    ## kin, pot, pop, coord, hop in the main dictionary.
    input_val=[]
    for n,f in enumerate(files):
        input_val.append([n,f])
    kin=[[] for x in range(len(input_val))]
    pot=[[] for x in range(len(input_val))]
    pop=[[] for x in range(len(input_val))]
    label=[[] for x in range(len(input_val))]
    coord=[[] for x in range(len(input_val))]
    hop=[[] for x in range(len(input_val))]

    if (len(input_val)) < procs:
        procs=len(input_val)
    sys.stdout.write('CPU: %3d Reading data: \r' % (procs))
    pool=multiprocessing.Pool(processes=procs)
    for ntraj, val in enumerate(pool.imap_unordered(read_func[prog],input_val)):
        ntraj+=1
        p,natom,nstate,nstep,dtime,trj_kin,trj_pot,trj_pop,trj_lab,trj_coord,trj_hop,crt_state=val

        ## create a list to classify trajectories according to their final state
        if ntraj == 1:
            prod=[[] for i in range(nstate)]

        ## send data
        prod[crt_state].append(p+1)
        kin[p]=trj_kin
        pot[p]=trj_pot
        pop[p]=trj_pop
        label[p]=trj_lab
        coord[p]=trj_coord
        hop[p]=trj_hop
        sys.stdout.write('CPU: %3d Reading data: %6.2f%% %d/%d\r' % (procs,ntraj*100/(len(input_val)),ntraj,len(input_val)))

    ## sort trajectory indices
    prod=[sorted(i) for i in prod]

    length=[]
    for p in pop:
        length.append(len(p))
    nstep = int(np.amax(length))

    main_dict={
    'title':  title,
    'natom':  natom,
    'nstate': nstate,
    'ntraj':  ntraj,
    'nstep':  nstep,
    'dtime':  dtime,
    'kin':    kin,
    'pot':    pot,
    'pop':    pop,
    'label':  label,
    'coord':  coord,
    'hop':    hop,
    'prod':   prod
    }

    #print('title,natom,nstate,ntraj,dtime')
    #print(title,natom,nstate,ntraj,dtime)
    #print('len(kin),len(pot),len(pop),len(coord)')
    #print(len(kin),len(pot),len(pop),len(coord))
    #print(kin,pot)
    #print('1',len(pot[0]),len(pop[0]),'1')
    #print(len(kin[0]),len(pot[0][0]),len(pop[0][0]),len(coord[0]))
    #exit()
    return main_dict

def read_raw_data_molcas(files):
    ## This function read data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj,f=files
    t=f.split('/')[-1]

    if os.path.exists('%s/%s.log' % (f,t)) == True:
        with open('%s/%s.log' % (f,t),'r') as logfile:
            log=logfile.read().splitlines()

    elif os.path.exists('%s/%s.out' % (f,t)) == True:
        with open('%s/%s.out' % (f,t),'r') as logfile:
            log=logfile.read().splitlines()

    with open('%s/%s.md.energies' % (f,t),'r') as engfile:
        eng=engfile.read().splitlines()

    with open('%s/%s.md.xyz' % (f,t),'r') as xyzfile:
        xyz=xyzfile.read().splitlines()

    ## Find population and label for each structure
    trj_pop=[]    # population list
    trj_lab=[]    # label list
    trj_hop=[]    # hopping event list
    crt_pop=''    # current population
    crt_label=''  # current label
    crt_hop=0     # current index for hopping event
    crt_state=0   # current state
    ini_state=0   # initial state
    pmd = 0       # flag to read pyrai2md log
    n=0
    i=0
    version = 19
    for line in log:
        n+=1
        if 'version:' in line:
            try:
                version = float(line.split()[-1])
            except:
                version = 18

        if 'PyRAI2MD' in line:
            pmd = 1

        if 'Root chosen for geometry opt' in line:
            i+=1
            crt_state=int(line.split()[-1])-1
       	    if i == 1:
       	       	ini_state=crt_state
            crt_label='traj %d coord %d init %s state %s' % (ntraj+1,i,ini_state,crt_state)

        if 'Gnuplot' in line:
            crt_hop+=1   # it is more reliable to count crt_hop separately, although it equals to i if the job complete normally
            if version > 18 or pmd == 1:
                event_checker=log[n+3]
                event_info=log[n+5]
                num_state=int((len(line.split())-2)/2)
                crt_pop=line.split()[1:num_state+1]
                crt_pop=[float(i) for i in crt_pop]
            else:
                event_checker=log[n+4]
       	       	event_info=log[n+6]
                crt_pop=line.split()[1:]
                crt_pop=[float(i) for i in crt_pop]

            if 'event' in event_checker:
                hop_state=int(event_info.split()[-2])
                crt_label+=' to %d CI' % (hop_state-1)
                trj_hop.append(crt_hop)

            trj_lab.append(crt_label)
            trj_pop.append(crt_pop)
            if i == mstep and mstep != 0: # cutoff trajectories, mstep is the global variable of cutoff step
                break

    nstate=len(trj_pop[0])    
    ## Find kinetic and potential energy for each state and time step
    trj_kin=[]    # kinetic energy list
    trj_pot=[]    # potential energy list
    trj_tim=[]    # time step list
    crt_time=''   # current time step
    crt_kin=''    # current kinetic energy
    crt_pot=''    # current potential energy
    n=0
    for line in eng:
        n+=1
        if 'time' in line:
            continue # skip the title line
        line=line.replace('D','e')
        line=line.split()
        crt_time=float(line[0])
        crt_kin=float(line[2])
        crt_pot=line[4:nstate+4]
        crt_pot=[float(i) for i in crt_pot]
        trj_kin.append(crt_kin)
        trj_pot.append(crt_pot)
        trj_tim.append(crt_time)
       	if n-1 == mstep and mstep != 0: # cutoff trajectories
       	    break

    dtime=trj_tim[1]
    ## Find coordinates 
    trj_coord=[]   # coordinates list
    crt_coord=''   # current coordinates
    natom=int(xyz[0])    
    n=0            # count line number
    m=0            # count structure number
    for line in xyz:
        n+=1
        if n % (natom+2) == 0: # at the last line of each coordinates
            m+=1
            crt_coord=[trj_lab[m-1]]+xyz[n-natom:n] # combine label with coordinates 
            trj_coord.append(crt_coord)
            if m == mstep and mstep != 0: # cutoff trajectories
                break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab=len(trj_lab)
    len_pop=len(trj_pop)
    len_kin=len(trj_kin)
    len_pot=len(trj_pot)
    len_crd=len(trj_coord)
    nstep=np.amin([len_lab,len_pop,len_kin,len_pot,len_crd]).tolist()

    trj_lab=trj_lab[0:nstep]
    trj_pop=trj_pop[0:nstep]
    trj_kin=trj_kin[0:nstep]
    trj_pot=trj_pot[0:nstep]
    trj_coord=trj_coord[0:nstep]

    return ntraj,natom,nstate,nstep,dtime,trj_kin,trj_pot,trj_pop,trj_lab,trj_coord,trj_hop,crt_state

def read_raw_data_newtonx(files):
    ## This function read newtonx data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj,f=files

    with open('%s/RESULTS/nx.log' % (f),'r') as nxfile:
        nx=nxfile.read().splitlines()

    with open('%s/RESULTS/dyn.out' % (f),'r') as logfile:
        log=logfile.read().splitlines()

    with open('%s/RESULTS/sh.out' % (f),'r') as popfile:
        pop=popfile.read().splitlines()

    with open('%s/RESULTS/en.dat' % (f),'r') as engfile:
        eng=engfile.read().splitlines()

    ## Find label and coordinates for each structure
    natom=0       # number of atoms
    nstate=0      # number of states
    trj_lab=[]    # label list
    trj_hop=[]    # hopping event list
    trj_coord=[]  # coordinates list
    crt_label=''  # current label
    crt_hop=0     # current index for hopping event
    crt_coord=''  # current coordinates
    crt_state=0   # current state
    ini_state=0   # initial state
    pre_state=0   # previous state
    n=0
    i=0
    for line in nx:
        if 'Nat' in line:
            natom=int(line.split()[-1])
        if 'nstat ' in line:
            nstate=int(line.split()[-1])
        if 'etot_drift' in line:
            break

    for line in log:
        n+=1
        if 'Molecular dynamics on state' in line:
            i+=1
            crt_state=int(line.split()[6])-1
       	    if i == 1:
       	       	ini_state=crt_state
                pre_state=crt_state

            crt_label='traj %d coord %d init %s' % (ntraj+1,i,ini_state)

            if pre_state != crt_state:
                crt_hop=i
                hop_state=crt_state
                crt_label+=' state %d to %d CI' % (pre_state,hop_state)
                trj_hop.append(crt_hop)
                pre_state=crt_state
            else:
                crt_label+=' state %d' % (crt_state)

            xyz=log[n-1+4:n-1+natom+4] # find coordinates
            xyz=['%-5s %14.8f %14.8f %14.8f' % (x.split()[0],float(x.split()[2])*0.529177,float(x.split()[3])*0.529177,float(x.split()[4])*0.529177) for x in xyz] # pick atom,x,y,z
            crt_coord=[crt_label]+xyz # combine label with coordinates

            trj_lab.append(crt_label)
            trj_coord.append(crt_coord)

            if i == mstep and mstep != 0: # cutoff trajectories
                break
    trj_lab=trj_lab[:-1]    
    trj_coord=trj_coord[:-1] # newtonx does not compute the last geometry

    ## Find state population for each structure
    trj_pop=[]    # population list
    crt_pop=''    # current population
    record=0
    n=0
    i=0
    for line in pop:
        n+=1
        if '|v.h|' in line:
            i+=1
        
        if len(line.split()) == 5 and 'POTENTIAL ENERGY VARIATION' not in line:
            if int(line.split()[1]) == i-1:
                crt_pop=pop[n-1:n-1+nstate]
                crt_pop[0]=crt_pop[0].split()[3]
                crt_pop=[float(i) for i in crt_pop]
                trj_pop.append(crt_pop)

            if i == mstep and mstep != 0: # cutoff trajectories
                break

    ## Find kinetic and potential energy for each state and time step
    trj_kin=[]    # kinetic energy list
    trj_pot=[]    # potential energy list
    trj_tim=[]    # time step list
    crt_time=''   # current time step
    crt_kin=''    # current kinetic energy
    crt_pot=''    # current potential energy
    n=0
    for line in eng:
        n+=1
        line=line.split()
        crt_time=float(line[0])
        crt_kin=float(line[-1])-float(line[-2])
        crt_pot=line[1:nstate+1]
        crt_pot=[float(i) for i in crt_pot]
        trj_kin.append(crt_kin)
        trj_pot.append(crt_pot)
        trj_tim.append(crt_time)

       	if n-1 == mstep and mstep != 0: # cutoff trajectories
       	    break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab=len(trj_lab)
    len_pop=len(trj_pop)
    len_kin=len(trj_kin)
    len_pot=len(trj_pot)
    len_crd=len(trj_coord)
    nstep=np.amin([len_lab,len_pop,len_kin,len_pot,len_crd]).tolist()

    trj_lab=trj_lab[0:nstep]
    trj_pop=trj_pop[0:nstep]
    trj_kin=trj_kin[0:nstep]
    trj_pot=trj_pot[0:nstep]
    trj_coord=trj_coord[0:nstep]

    dtime=trj_tim[1]

    return ntraj,natom,nstate,nstep,dtime,trj_kin,trj_pot,trj_pop,trj_lab,trj_coord,trj_hop,crt_state

def read_raw_data_sharc(files):
    ## This function read sharc data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj,f=files

    with open('%s/output.dat' % (f),'r') as datfile:
        dat=datfile.read().splitlines()

    ## Find label and coordinates for each structure
    natom=0	  # number of atoms
    nstate=0	  # number of states
    trj_lab=[]    # label list
    trj_hop=[]    # hopping event list
    trj_coord=[]  # coordinates list
    trj_pop=[]    # population list
    trj_kin=[]    # kinetic energy list
    trj_pot=[]    # potential energy list
    trj_state=[]  # state list
    crt_label=''  # current label
    crt_hop=0     # current index for hopping event
    crt_coord=''  # current coordinates
    crt_pop=''    # current population
    crt_kin=''    # current kinetic energy
    crt_pot=''    # current potential energy
    crt_state=0   # current state
    ini_state=0   # initial state
    pre_state=0   # previous state
    zero_energy = 0 # sharc ref energy
    i=0
    for n, line in enumerate(dat):
        if 'natom' in line:
            natom = int(line.split()[-1])
        if 'nstates_m ' in line:
            nstate = np.sum([(n + 1) * int(x) for n,x in enumerate(line.split()[1:])])
            nstate = int(nstate)
        if 'dtstep' in line:
            dtime = float(line.split()[-1]) / 20.6706868947804 * 0.5
        if 'ezero' in line:
            zero_energy=float(line.split()[-1])
        if '! Elements' in line:
            atom = dat[n + 1: n + 1 + natom]

        if '! 1 Hamiltonian' in line:
            i += 1
            crt_pot = dat[n + 1: n + 1 + nstate]
            crt_pot = np.array([p.split() for p in crt_pot]).astype(float)[:,::2]
            crt_pot += zero_energy
            crt_pot = np.diag(crt_pot).tolist()
            trj_pot.append(crt_pot)

        if '! 5 Coefficients' in line:
            crt_pop = dat[n + 1: n + 1 + nstate]
            crt_pop = np.array([p.split() for p in crt_pop]).astype(float)
            crt_pop = np.sum(crt_pop**2, axis = 1).tolist()
            trj_pop.append(crt_pop)

        if '! 7 Ekin' in line:
            crt_kin = float(dat[n + 1])
            trj_kin.append(crt_kin)

        if '! 8 states (diag, MCH)' in line:
            crt_state = int(dat[n + 1].split()[-1]) - 1
            trj_state.append(crt_state)
            if i == 1:
                ini_state = crt_state
                pre_state = crt_state
            crt_label='traj %d coord %d init %s' % (ntraj+1,i,ini_state)
            if pre_state != crt_state:
                crt_hop = i
                hop_state = crt_state
                crt_label += ' state %d to %d CI' % (pre_state,hop_state)
                trj_hop.append(crt_hop)
                pre_state = crt_state
            else:
                crt_label += ' state %d' % (crt_state)
            trj_lab.append(crt_label)

        if '! 11 Geometry in a.u.' in line:
            xyz = dat[n + 1: n + 1 + natom]
            xyz = ['%-5s %14.8f %14.8f %14.8f' % (atom[n],float(x.split()[0])*0.529177,float(x.split()[1])*0.529177,float(x.split()[2])*0.529177) for n, x in enumerate(xyz)] # pick atom,x,y,z
            crt_coord=[crt_label]+xyz # combine label with coordinates
            trj_coord.append(crt_coord)

        if i == mstep and mstep != 0: # cutoff trajectories
            break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab=len(trj_lab)
    len_pop=len(trj_pop)
    len_kin=len(trj_kin)
    len_pot=len(trj_pot)
    len_crd=len(trj_coord)
    nstep=np.amin([len_lab,len_pop,len_kin,len_pot,len_crd]).tolist()

    trj_lab=trj_lab[0:nstep]
    trj_pop=trj_pop[0:nstep]
    trj_kin=trj_kin[0:nstep]
    trj_pot=trj_pot[0:nstep]
    trj_coord=trj_coord[0:nstep]

    return ntraj,natom,nstate,nstep,dtime,trj_kin,trj_pot,trj_pop,trj_lab,trj_coord,trj_hop,crt_state

def read_raw_data_fromage(files):
    ## This function read fromage data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj,f=files

    with open('%s/output.chk' % (f), 'r') as datfile:
        dat = datfile.read().splitlines()

    with open('%s/geom_mol.xyz' % (f), 'r') as xyzfile:
        xyz = xyzfile.read().splitlines()

    ## Find label and coordinates for each structure
    natom=0	  # number of atoms
    nstate=0	  # number of states
    trj_lab=[]    # label list
    trj_hop=[]    # hopping event list
    trj_coord=[]  # coordinates list
    trj_pop=[]    # population list
    trj_kin=[]    # kinetic energy list
    trj_pot=[]    # potential energy list
    trj_state=[]  # state list
    crt_label=''  # current label
    crt_hop=0     # current index for hopping event
    crt_coord=''  # current coordinates
    crt_pop=''    # current population
    crt_kin=''    # current kinetic energy
    crt_pot=''    # current potential energy
    crt_state=0   # current state
    ini_state=0   # initial state
    pre_state=0   # previous state
    i=0

    ## Find energies and population
    checkline = dat[4].split()
    dtime = float(checkline[1])
    nstate = int(len(checkline[5:])/2)

    for n, line in enumerate(dat[3:]):
        line = line.split()
        crt_state = int(line[2]) - 1

        if n == 0:
            ini_state = crt_state
            pre_state = crt_state
        crt_label='traj %d coord %d init %s' % (ntraj + 1, n + 1, ini_state)

        if pre_state != crt_state:
            crt_hop = n + 1
            hop_state = crt_state
            crt_label += ' state %d to %d CI' % (pre_state, hop_state)
            trj_hop.append(crt_hop)
            pre_state = crt_state
        else:
            crt_label += ' state %d' % (crt_state)
        trj_lab.append(crt_label)

        crt_kin = float(line[4])
        trj_kin.append(crt_kin)

        crt_pot = line[6: nstate + 6]
        crt_pot = [float(x) for x in crt_pot]
        trj_pot.append(crt_pot)

        crt_pop = line[nstate + 6: nstate + nstate + 6]
        crt_pop = [float(x) for x in crt_pop]
        trj_pop.append(crt_pop)

        if n + 1 == mstep and mstep != 0: # cutoff trajectories
            break

    ## Find coordiantes

    natom = int(xyz[0])
    n = 0            # count line number
    m = 0            # count structure number

    for line in xyz:
        n += 1
        if n % (natom + 2) == 0: # at the last line of each coordinates
            m += 1
            crt_coord = [trj_lab[m - 1]] + xyz[n - natom: n] # combine label with coordinates 
            trj_coord.append(crt_coord)
            if m == mstep and mstep != 0: # cutoff trajectories
                break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab=len(trj_lab)
    len_pop=len(trj_pop)
    len_kin=len(trj_kin)
    len_pot=len(trj_pot)
    len_crd=len(trj_coord)
    nstep=np.amin([len_lab,len_pop,len_kin,len_pot,len_crd]).tolist()

    trj_lab=trj_lab[0:nstep]
    trj_pop=trj_pop[0:nstep]
    trj_kin=trj_kin[0:nstep]
    trj_pot=trj_pot[0:nstep]
    trj_coord=trj_coord[0:nstep]

    return ntraj,natom,nstate,nstep,dtime,trj_kin,trj_pot,trj_pop,trj_lab,trj_coord,trj_hop,crt_state

def HOPprod(raw_data,follow,product):
    ## This function analyze products 
    ## This function unpack main dictionary from raw_data
    ## This function save sorted products in Prod.state.xyz and CI in CI.state.xyz 
    ## This function save state, kinetic and potential energy, population in .dat and label coordinates in .xyz when follow != None

    title=raw_data['title']
    natom=raw_data['natom']
    nstate=raw_data['nstate']
    ntraj=raw_data['ntraj']
    dtime=raw_data['dtime']
    kin=raw_data['kin']
    pot=raw_data['pot']
    pop=raw_data['pop']
    label=raw_data['label']
    coord=raw_data['coord']
    hop=raw_data['hop']
    prod=raw_data['prod']

    state_info=''
    trajs_info=''
    ## output CI and products coordinates according to final state
    print('\nNumber of Trajectories: %5d\nNumber of States:       %5d\n' % (ntraj,nstate))
    for i in range(nstate):        #loop over states
        products=''
        conical=''
        for j in prod[i]:          # trajectory index in each state
            products+=format(natom,coord[j-1][-1])
            if len(hop[j-1]) == 0:
                continue
            for k in hop[j-1]:     # CI index in each trajectory
                if k <= len(coord[j-1]):
                    conical+=format(natom,coord[j-1][k-1])

        with open('Prod.S%d.xyz' % (i),'w') as outxyz:
            outxyz.write(products)
        with open('CI.S%d.xyz'   % (i),'w') as outxyz:
       	    outxyz.write(conical)
        index_range=redindex(prod[i])
        print('\nState %5d:  %5d  Write: Prod.S%d.xyz and CI.S%d.xyz Range: %s' % (i,len(prod[i]),i,i,index_range))
        state_info+='\nState %5d:  %5d Write: Prod.S%d.xyz and CI.S%d.xyz\n\n%s\n' % (i,len(prod[i]),i,i,format2(index_range))
    ## output data for selected trajectory
    if follow != None:
        print('\nSelect trajectory: %s' % (redindex(follow)))
        trajs_info+='\nSelect trajectory:\n\n%s\n' % (format2(redindex(follow)))
        for i in follow:
            print('Traj: %5d Write: select-%s-%s.xyz and select-%s-%s.dat' % (i,title,i,title,i))
            trajs_info+='Traj: %5d Write: select-%s-%s.xyz and select-%s-%s.dat' % (i,title,i,title,i)
            select=''
            for j in coord[i-1]:
                select+=format(natom,j)
            with open('select-%s-%s.xyz' % (title,i),'w') as outxyz:
                outxyz.write(select)
            cap_pot=''.join(['%24s' % ('Epot %s' % (l)) for l in range(nstate)])
            cap_pop=''.join(['%24s' % ('Pop %s'  % (l)) for l in range(nstate)])
            select='%-20d%5s%24s%s%s\n' % (dtime,'state','Ekin',cap_pot,cap_pop)
            for n,k in enumerate(kin[i-1]):
                crt_label=label[i-1][n]
                crt_state=crt_label.split()[7]## state number is the 8th data
                crt_state=int(crt_state)
                crt_pot=np.array(pot[i-1])
                crt_pop=np.array(pop[i-1])
                cap_pot=''.join(['%24.16f' % (x) for x in crt_pot[n]])
                cap_pop=''.join(['%24.16f' % (x) for x in crt_pop[n]])
                select+='%-20d%5d%24.16f%s%s\n' % (n,crt_state,k,cap_pot,cap_pop)

            with open('select-%s-%s.dat' % (title,i),'w') as outxyz:
                outxyz.write(select)

    if product != None:
        products=''
        for j in product:          # trajectory index in each state
            products+=format(natom,coord[j-1][-1])

        with open('select-prod-%s.xyz' % (title),'w') as outxyz:
                outxyz.write(products)

        print('\nWrite: select-prod-%s.xyz Range: %s' % (title,redindex(product)))
        trajs_info+='\nWrite: select-prod-%s.xyz\n%s\n' % (title,format2(redindex(product)))

    log_info="""
=> Product quick analysis
-------------------------------------
Number of atoms:            %-10s
Number of states:           %-10s
Number of trajectories:     %-10s
Time step (a.u.):           %-10s
%s
%s

    """ % (natom,nstate,ntraj,dtime,state_info,trajs_info)

    with open('%s.traj.log' % (title),'a') as trajlog:
        trajlog.write(log_info)

def HOPtraj(raw_data,product,follow,ref,track,diff,save):
    ## This function analyze trajectories
    ## This function unpack main dictionary from raw_data
    ## This function compare product by paramters in track
    ## This function call compute_para for parallelization
    ## This function save averaged kinetic and potential energy, population in avg-name.dat when follow != None
    ## This function save trajectory analysis when follow != None

    title=raw_data['title']
    natom=raw_data['natom']
    nstate=raw_data['nstate']
    ntraj=raw_data['ntraj']
    nstep=raw_data['nstep']
    dtime=raw_data['dtime']
    kin=raw_data['kin']
    pot=raw_data['pot']
    pop=raw_data['pop']
    label=raw_data['label']
    coord=raw_data['coord']
    hop=raw_data['hop']
    prod=raw_data['prod']
    par_track,par_std=Trkread(track)
    par_name,par_cmmt=Pararead(par_track)
    ref_coord=Refread(natom,ref)

    ## schedule number of trajectory for analysis
    ## ground_state is for product analysis
    ## average_traj is for average analysis
    if product != None:
        ground_state=product
        average_traj=product
    else:
        ground_state=prod[0]
        average_traj=range(1,ntraj+1)

    ## product analysis
    ref_par=[[y for y in par_track] for x in ref_coord]
    prd_par=[[y for y in par_track] for x in ground_state]
    dev_par=[[[z for z in par_track] for y in ref_coord] for x in ground_state]

    if mstep != 0:
        nstep=mstep

    ## compare products with reference
    input_val=[]
    for n,i in enumerate(ground_state):
        i=coord[i-1][-1][1:]            # take the last structure of traj i and remove the label
        for m,j in enumerate(ref_coord):
            for l,k in enumerate(par_track):
                 input_val.append([0,n,m,l,i,j,k,1]) # fake traj index(0), (n)th structure(i), (m)th reference structure(j), (l)th parameter(k), compute the difference(1)

    procs=cpus   # get global cpus
    if (len(input_val)) < procs:
        procs=len(input_val)
    sys.stdout.write('CPU: %3d Product analysis: \r' % (procs))
    pool=multiprocessing.Pool(processes=procs)
    for p, val in enumerate(pool.imap_unordered(compute_para,input_val)):
        p+=1
        ref_par[val[2]][val[3]]=val[5]
        prd_par[val[1]][val[3]]=val[4]
        dev_par[val[1]][val[2]][val[3]]=val[6]
        sys.stdout.write('CPU: %3d Product analysis: %6.2f%% %d/%d\r' % (procs,p*100/(len(input_val)),p,len(input_val) ))
    ## print results
    ref_info='\n\nReference structure\n%10s%s\n%10s%s\n\n' % ('Ref',''.join(['%14s' % (x) for x in par_name]),'Cmt',''.join(['%14s' % (x) for x in par_cmmt]))
    for n,i in enumerate(ref_par):
        ref_info+='%5d%5s%s\n' % (n+1,'',''.join(['%14.4f' % (x) for x in i]))
    print(ref_info)
    prd_info='\nLast snapshot structure\n%5s%5s%s\n%10s%s\n\n' % ('Index','Traj',''.join(['%14s' % (x) for x in par_name]),'Comment',''.join(['%14s' % (x) for x in par_cmmt]))
    for n,i in enumerate(prd_par):
       	prd_info+='%5d%5d%s\n' % (n+1,ground_state[n],''.join(['%14.4f' % (x) for x in i]))
    print(prd_info)

    dev_info='\n%10s%s\n' % ('Comparison',''.join((['%11s   ' % (x+1) for x in range(len(par_name))]+['|'])*len(ref_coord)))
    dev_info+='%5s%5s%s\n' % ('Index','Traj',''.join( (['%14s' % (x) for x in par_name]+['|'])*len(ref_coord) ))
    dev_info+='%10s%s\n\n' % ('Comment',''.join( (['%14s' % (x) for x in par_cmmt]+['|'])*len(ref_coord) ))
    for n,i in enumerate(dev_par):
        row=''
        for j in i:
           row+=''.join(['%14.4f' % (x) for x in j])
           row+=' '
        dev_info+='%5d%5d%s\n' % (n+1,ground_state[n],row)
    print(dev_info)

    ## extract products with deviation under the threshold
    productive=[]
    nonproductive=[]
    rmsd_info=''
    if par_std > -1: 
        print('Choose parameter: %5d Threshold: %14.4f' % (par_std,thrhd))
        rmsd_info+='Choose parameter: %5d Threshold: %14.4f\n' % (par_std,thrhd)
        for n,i in enumerate(dev_par):
            if i[0][par_std-1] <= thrhd: ## only compare the first reference structure
                productive.append(ground_state[n])
            else:
                nonproductive.append(ground_state[n])

        if len(nonproductive) == 0:
            nonproductive='None'
            print('Different trajecotry: %s' % (nonproductive))
            rmsd_info+='Different trajecotry: %s\n' % (nonproductive)
        else:
            print('Different trajecotry: %5d\n\n %s\n' % (len(nonproductive),redindex(nonproductive)))
            rmsd_info+='Different trajecotry: %5d Range:\n %s\n' % (len(nonproductive),format2(redindex(nonproductive)))

        if len(productive) == 0:
            productive='None'
            print('Similar trajecotry: %s' % (productive))
            rmsd_info+='Similar trajecotry: %s\n' % (productive)
        else:
            print('Similar trajecotry: %5d\n\n %s\n' % (len(productive),redindex(productive)))
            rmsd_info+='Similar trajecotry: %5d Range:\n %s\n' % (len(productive),format2(redindex(productive)))

    ## average analysis
    ## average kinetic energy, potential energy, and population for productive trajectories
    if save >= 1:
        repair=[]
        all_lab=[]
        all_kin=[]
        all_pot=[]
        all_tot=[]
        all_pop=[]
        avg_kin=np.zeros(nstep)
        avg_pot=np.zeros([nstep,nstate])
        avg_tot=np.zeros([nstep,nstate])
        avg_pop=np.zeros([nstep,nstate])
        for a in average_traj:
            crt_lab=[x.split()[7] for x in label[a-1]] ## state number is the 8th data
            crt_kin=kin[a-1]
            crt_pot=pot[a-1]
            crt_pop=pop[a-1]
            crt_tot=(np.array([[x for y in range(nstate)] for x in kin[a-1]])+np.array(crt_pot)).tolist() ## expand kinetic energy to each state

            ## compute the missing step
            dstep=nstep-len(kin[a-1])
            if dstep != 0:    # complete the missing part by repeating the last step
                repair.append(a)
                crt_lab=crt_lab+[crt_lab[-1] for x in range(dstep)]
                crt_kin=crt_kin+[crt_kin[-1] for x in range(dstep)]
                crt_pot=crt_pot+[crt_pot[-1] for x in range(dstep)]
                crt_tot=crt_tot+[crt_tot[-1] for x in range(dstep)]
                crt_pop=crt_pop+[crt_pop[-1] for x in range(dstep)]

            avg_kin+=np.array(crt_kin)
            avg_pot+=np.array(crt_pot)
            avg_tot+=np.array(crt_tot)
            avg_pop+=np.array(crt_pop)

            all_lab.append(crt_lab)
            all_kin.append(crt_kin)
            all_pot.append(crt_pot)
            all_tot.append(crt_tot)
            all_pop.append(crt_pop)

        avg_kin/=len(average_traj)
        avg_pot/=len(average_traj)
        avg_tot/=len(average_traj)
        avg_pop/=len(average_traj)

        cap_kin='%24s' % ('Ekin')
        cap_pot=''.join(['%24s' % ('Epot %s' % (l)) for l in range(nstate)])
        cap_tot=''.join(['%24s' % ('Etot %s' % (l)) for l in range(nstate)])
        cap_pop=''.join(['%24s' % ('Pop %s'  % (l)) for l in range(nstate)])
        average='%-20d%s%s%s%s\n' % (dtime,cap_kin,cap_pot,cap_tot,cap_pop)
        for a in range(nstep):
            cap_pot=''.join(['%24.16f' % (x) for x in avg_tot[a]])
            cap_tot=''.join(['%24.16f' % (x) for x in avg_tot[a]])
       	    cap_pop=''.join(['%24.16f' % (x) for x in avg_pop[a]])
            average+='%-20d%24.16f%s%s%s\n' % (a,avg_kin[a],cap_pot,cap_tot,cap_pop)

        average_info=''
        average_info+='Repair trajectories:   %5d of %5d\n\n%s\n' % (len(repair),len(average_traj),format2(redindex(repair)))
        average_info+='Averaged trajectories: %5d\n' % (len(average_traj))
        average_info+='Save summary:            average-%s.dat\n'               % (title)

        print(average_info)
        rmsd_info+=average_info

        with open('average-%s.dat' % (title),'w') as outsum:
            outsum.write(average)

        if save >= 2:
            average_info=''
            average_info+='Save state populations:  state-pop-%s.json\n'        % (title)
            average_info+='Save energy profiles:    energy-profile-%s.json\n'    % (title)

            print(average_info)
            rmsd_info+=average_info

            with open('state-pop-%s.json' % (title), 'w') as outpop:
                json.dump(all_pop,outpop)
            with open('energy-profile-%s.json' % (title), 'w') as outtot:
                json.dump([all_lab,all_kin,all_pot,all_tot],outtot)

    ## output parameters for selected trajectories
    traj_info=''
    if follow != None:
        traj_info+='\nSelect trajectory:\n\n %s\n' % (format2(redindex(follow)))
        print('\nSelect trajectory: %s\n' % (redindex(follow)))
        plot_para=[]
        plot_Etot=[]
        plot_type=[]
        plot_hop=[]
        plot_label=[]
        input_val=[]
        if productive == 'None':                         ## convert productive back to an empty list
            prodctive = []
        for o,h in enumerate(follow):                    ## for each selected trajectory
            trj_data=[]
            trj_Etot=[]
            if h in productive:                          ## classify trajectories according to if they formed the expected product in reference
                plot_type.append(1)
            else:
                plot_type.append(0)
            
            plot_hop.append(hop[h-1])
            plot_label.append(label[h-1])
            for n,i in enumerate(coord[h-1]):            ## for each coordinates
                i=i[1:]                                  ## remove the label
                crt_label=label[h-1][n]
                crt_state=crt_label.split()[7]           ## state number is the 8th data
                crt_state=int(crt_state)
                crt_kin=kin[h-1][n]
                crt_pot=pot[h-1][n][crt_state]
                crt_tot=crt_kin+crt_pot
                str_data=[]
                trj_Etot.append(crt_tot)
                for m,j in enumerate(ref_coord):         ## for each reference coordinates
                    ref_data=[]
                    for l,k in enumerate(par_track):     ## for each parameters
                        prd_val,ref_val,dev_val=0,0,0    ## create fake data
                        input_val.append([o,n,m,l,i,j,k,diff])

                        ## skip the reference and difference to save space 
                        if diff == 0:
                            ref_data.append([prd_val])
                        else:
                            ref_data.append([prd_val,ref_val,dev_val])

                    str_data.append(ref_data)
                trj_data.append(str_data)
            plot_para.append(trj_data)
            plot_Etot.append(trj_Etot)

        procs=cpus # get global cpus
        if (len(input_val)) < procs:
            procs=len(input_val)
        sys.stdout.write('CPU: %3d Saving plot data: \r' % (procs))
        pool=multiprocessing.Pool(processes=procs)
        for p, val in enumerate(pool.imap_unordered(compute_para,input_val)):
            p+=1
            plot_para[val[0]][val[1]][val[2]][val[3]][0]=val[4]
            if diff !=0:
                plot_para[val[0]][val[1]][val[2]][val[3]][1]=val[5]
                plot_para[val[0]][val[1]][val[2]][val[3]][2]=val[6]
            sys.stdout.write('CPU: %3d Saving plot data: %6.2f%% %d/%d\r' % (procs,p*100/(len(input_val)),p,len(input_val) ))
        plot_data=[par_name,par_cmmt,plot_para,plot_Etot,plot_type,plot_hop,plot_label]
        with open('track-%s.json' % (title),'w') as outxyz:
            print('\nPlot data saved in track-%s.json' % (title))
            json.dump(plot_data,outxyz)
        traj_info+='\nPlot data saved in track-%s.json' % (title)

    prod_info=ref_info+prd_info+dev_info+rmsd_info
    log_info="""
=> Product trajectory analysis
-------------------------------------
Number of CPUs:             %-10s
Number of atoms:            %-10s
Number of states:           %-10s
Number of trajectories:     %-10s
Time step (a.u.):           %-10s
%s
%s

    """ % (procs,natom,nstate,ntraj,dtime,prod_info,traj_info)

    with open('%s.traj.log' % (title),'a') as trajlog:
        trajlog.write(log_info)

def compute_para(var):
    ## This function wrap the evaluate for parallelzation
    o,n,m,l,i,j,k,diff=var
    i,j,k=evaluate(i,j,k,diff)
    return o,n,m,l,i,j,k

def main():
    ##  This is the main function.
    ##  It read all options from command line or control file and 
    ##  pass them to other module for sampling and generating structures.
    ##  The parser code has clear description for all used variables.
    ##  This function calls getindex to read index

    print('')
    usage="""

    FSSH Analyzer

    Usage:
      python3 HOP-FSSH.py -x control
      python3 HOP-FSSH.py -h for help

    Control keywords

      cpus       2
      read       1
      title      name of calculation files
      index      1 [single,range,separate range]  
      nstep    	 0
      mode       1
      print      1
      ref        name of the reference xyz file [if has multiple structures the first one is the major product]
      track      list of parameter to track the trajectory
      diff       0
      save       0

    For more information, please see Readme-HOP-FSSH.txt

    """
    description=''
    #analysis control
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-x', dest='extnl',    type=str,   nargs=1, help='A text file that contains control parameters.')

    global thrhd
    global mstep
    global cpus

    (options, args) = parser.parse_args()

    ## Default settings    
    cpus=1                 # Number of CPU for analysis
    opt_read=1             # Read from calculation folders (1) or JSON file (2) or run diagnosis of data (3)
    title=None             # Name of calculation files
    index='1'              # Index of calculation files
    mstep=0                # Maximum step per trajectory
    opt_mode=0             # Analysis mode. Skip analysis (0), analyze products and trajectories (1), only analyze products (2), only analyze trajectories (3)
    product=None           # Select trajectories to analyze products
    follow=None            # Select trajectories to output energy, coordinates, and analyze trajectories
    ref=None               # A reference coordinates file that has one or more structures
    track=None             # Paramters to track trajectories. Options and a file of them are acceptable
    thrhd=0.3              # Threshold of RMSD between product and reference
    diff=0                 # Compute the structural parameter difference between trajectory and reference
    save=0                 # Save detieled kinetic energy, potential energy, total energy, state population
    extnl=options.extnl    # A text file that contains control parameters
    prog='molcas'          # Output file format
    savejson = 1           # save trajectory data into json

    if   extnl != None:   #read controls if provided
        with open(extnl,'r') as ext:
            for line in ext:
                if len(line.split()) < 2:
                    continue
                key=line.split()[0].lower()
                if   'cpus'      == key:
                    cpus=int(line.split()[1])
                elif 'read'      == key:
                    opt_read=int(line.split()[1])
                elif 'title'     == key:
                    title=line.split()[1]
                elif 'index'     == key:
                    index=line.split()[1:]
                elif 'step'      == key:
                    mstep=int(line.split()[1])
                elif 'mode'      == key:
                    opt_mode=int(line.split()[1])
                elif 'prod'      == key:
                    product=line.split()[1:]
                elif 'print'     == key:
                    follow=line.split()[1:]
                elif 'ref'       == key:
                    ref=line.split()[1]
                elif 'track'     == key:
                    track=line.split()[1:]
                elif 'threshold' == key:
                    thrhd=float(line.split()[1])
                elif 'diff' == key:
                    diff=int(line.split()[1])
                elif 'save' == key:
                    save=int(line.split()[1])
                elif 'prog' == key:
                    prog=str(line.split()[1]).lower()
                elif 'json' == key:  
                    savejson=int(line.split()[1])

    if title == None:
        print ('\n!!! Cannot recognize name of calculations !!!\n')
        print (usage)
        print ('\n!!! Cannot recognize name of calculations !!!\n')
        exit()

    print('-------------------------------------')
    ## prepare index, print, and track
    if index == None:
        print ('\n!!! Cannot recognize index of calculations !!!\n')
        print (usage)
        print ('\n!!! Cannot recognize index of calculations !!!\n')
        exit()
    else:
        if os.path.exists('%s' % (index[0])) == True:
            print('\nRead index from file: %s' % (index[0]))
            opt_index='%s' % (index[0])
            with open(index[0],'r') as para:
                files=para.read().split()
        else:
            print('\nRead index from control')
            opt_index='control'
            index=getindex(index)
            files=[]
            for i in index:
                files.append('%s-%s' % (title,i))

    if product != None:
        if os.path.exists('%s' % (product[0])) == True:
            print('\nCheck product from file: %s' % (product[0]))
            opt_product='%s' % (product[0])
            with open(product[0],'r') as para:
                product=para.read().split()
        else:
            print('\nCheck product from control')
            opt_product='control'
        product=getindex(product)
    else:
        print('\nCheck product: ground-state')
        opt_product='ground-state'

    if follow != None:
        if os.path.exists('%s' % (follow[0])) == True:
            print('\nFollow trajectory from file: %s' % (follow[0]))
            opt_follow='%s' % (follow[0])
            with open(follow[0],'r') as para:
                follow=para.read().split()
        else:
            print('\nFollow trajectroy from control')
            opt_follow='control'
        follow=getindex(follow)
    else:
        print('\nFollow trajectroy: None')
        opt_follow='None'

    if track != None:
        if os.path.exists('%s' % (track[0])) == True:
            print('\nRead track paramters from file: %s' % (track[0]))
            opt_track='%s' % (track[0])
            with open(track[0],'r') as para:
                track=para.read().split()
        else:
            print('\nRead track paramters from control')
            opt_track='control'
    else:
        print('\nRead track paramters: None')
        opt_track='None'

    if mstep != 0:
        print('\nCutoff trajectory: %s' % (mstep))

    log_info="""
      Molcas TSH Analyzer Log

-------------------------------------
Title:                      %-10s
Data from:                  %-10s
Read data:                  %-10s
Read index:                 %-10s
Read selected product:      %-10s
Read selected trajectories: %-10s
Read track parameter:       %-10s
Mode:                       %-10s
Cutoff trajecotry:          %-10s
-------------------------------------

    """ % (title,prog,opt_read,opt_index,opt_product,opt_follow,opt_track,opt_mode,mstep)

    with open('%s.traj.log' % (title),'w') as trajlog:
        trajlog.write(log_info)

    if   opt_read == 1:
        print('\nReading data from calculation folders')
        raw_data=HOPread(title,files,prog)
        if savejson == 1:
            with open('%s.json' % (title),'w') as indata:
                json.dump(raw_data,indata)
            print('\nWriting data into file: %s.json' % (title))
        else:
            print('\nSkip writing data file')

    elif opt_read == 2:
       	print('\nRead data from file: %s.json' % (title))
        with open('%s.json' % (title),'r') as indata:
            raw_data=json.load(indata)

    elif opt_read == 3:
        print('\nRun diagnosis of data from calculation folders')
        HOPdiag(files,prog)
        exit()
    else:
        print ('\n!!! Cannot recognize read mode. Use 1, 2 or 3!!!\n')
        print (usage)
       	print ('\n!!! Cannot recognize read mode. Use 1, 2 or 3 !!!\n')
        exit()

    print('-------------------------------------')

    if   opt_mode == 1:
        HOPprod(raw_data,follow,product)
        HOPtraj(raw_data,product,follow,ref,track,diff,save)
    elif opt_mode == 2:
        HOPprod(raw_data,follow,product)
    elif opt_mode == 3:
        HOPtraj(raw_data,product,follow,ref,track,diff,save)
    else:
       	print ('\n!!! Skip analysis !!!\n')
        exit()

if __name__ == '__main__':
    main()

