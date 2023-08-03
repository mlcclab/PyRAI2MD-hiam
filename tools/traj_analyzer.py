## ----------------------
## Trajectory analysis script - a script to automatically extract trajectory data
## ----------------------
##
## Gen-FSSH.py 2019-2022 Jingbai Li
## New version Oct 3 2022 Jingbai Li

import sys
import os
import json
import multiprocessing
import numpy as np
from numpy import linalg as la
from scipy.optimize import linear_sum_assignment


class Element:

    def __init__(self, name):
        periodic_table = {
            "HYDROGEN": "1", "H": "1", "1": "1",
            "HELIUM": "2", "He": "2", "2": "2", "HE": "2",
            "LITHIUM": "3", "Li": "3", "3": "3", "LI": "3",
            "BERYLLIUM": "4", "Be": "4", "4": "4", "BE": "4",
            "BORON": "5", "B": "5", "5": "5",
            "CARBON": "6", "C": "6", "6": "6",
            "NITROGEN": "7", "N": "7", "7": "7",
            "OXYGEN": "8", "O": "8", "8": "8",
            "FLUORINE": "9", "F": "9", "9": "9",
            "NEON": "10", "Ne": "10", "10": "10", "NE": "10",
            "SODIUM": "11", "Na": "11", "11": "11", "NA": "11",
            "MAGNESIUM": "12", "Mg": "12", "12": "12", "MG": "12",
            "ALUMINUM": "13", "Al": "13", "13": "13", "AL": "12",
            "SILICON": "14", "Si": "14", "14": "14", "SI": "14",
            "PHOSPHORUS": "15", "P": "15", "15": "15",
            "SULFUR": "16", "S": "16", "16": "16",
            "CHLORINE": "17", "Cl": "17", "17": "17", "CL": "17",
            "ARGON": "18", "Ar": "18", "18": "18", "AR": "18",
            "POTASSIUM": "19", "K": "19", "19": "19",
            "CALCIUM": "20", "Ca": "20", "20": "20", "CA": "20",
            "SCANDIUM": "21", "Sc": "21", "21": "21", "SC": "21",
            "TITANIUM": "22", "Ti": "22", "22": "22", "TI": "22",
            "VANADIUM": "23", "V": "23", "23": "23",
            "CHROMIUM": "24", "Cr": "24", "24": "24", "CR": "24",
            "MANGANESE": "25", "Mn": "25", "25": "25", "MN": "25",
            "IRON": "26", "Fe": "26", "26": "26", "FE": "26",
            "COBALT": "27", "Co": "27", "27": "27", "CO": "27",
            "NICKEL": "28", "Ni": "28", "28": "28", "NI": "28",
            "COPPER": "29", "Cu": "29", "29": "29", "CU": "29",
            "ZINC": "30", "Zn": "30", "30": "30", "ZN": "30",
            "GALLIUM": "31", "Ga": "31", "31": "31", "GA": "31",
            "GERMANIUM": "32", "Ge": "32", "32": "32", "GE": "32",
            "ARSENIC": "33", "As": "33", "33": "33", "AS": "33",
            "SELENIUM": "34", "Se": "34", "34": "34", "SE": "34",
            "BROMINE": "35", "Br": "35", "35": "35", "BR": "35",
            "KRYPTON": "36", "Kr": "36", "36": "36", "KR": "36",
            "RUBIDIUM": "37", "Rb": "37", "37": "37", "RB": "37",
            "STRONTIUM": "38", "Sr": "38", "38": "38", "SR": "38",
            "YTTRIUM": "39", "Y": "39", "39": "39",
            "ZIRCONIUM": "40", "Zr": "40", "40": "40", "ZR": "40",
            "NIOBIUM": "41", "Nb": "41", "41": "41", "NB": "41",
            "MOLYBDENUM": "42", "Mo": "42", "42": "42", "MO": "42",
            "TECHNETIUM": "43", "Tc": "43", "43": "43", "TC": "43",
            "RUTHENIUM": "44", "Ru": "44", "44": "44", "RU": "44",
            "RHODIUM": "45", "Rh": "45", "45": "45", "RH": "45",
            "PALLADIUM": "46", "Pd": "46", "46": "46", "PD": "46",
            "SILVER": "47", "Ag": "47", "47": "47",
            "CADMIUM": "48", "Cd": "48", "48": "48", "CD": "48",
            "INDIUM": "49", "In": "49", "49": "49", "IN": "49",
            "TIN": "50", "Sn": "50", "50": "50", "SN": "50",
            "ANTIMONY": "51", "Sb": "51", "51": "51", "SB": "51",
            "TELLURIUM": "52", "Te": "52", "52": "52", "TE": "52",
            "IODINE": "53", "I": "53", "53": "53",
            "XENON": "54", "Xe": "54", "54": "54", "XE": "54",
            "CESIUM": "55", "Cs": "55", "55": "55", "CS": "55",
            "BARIUM": "56", "Ba": "56", "56": "56", "BA": "56",
            "LANTHANUM": "57", "La": "57", "57": "57", "LA": "57",
            "CERIUM": "58", "Ce": "58", "58": "58", "CE": "58",
            "PRASEODYMIUM": "59", "Pr": "59", "59": "59", "PR": "59",
            "NEODYMIUM": "60", "Nd": "60", "60": "60", "ND": "60",
            "PROMETHIUM": "61", "Pm": "61", "61": "61", "PM": "61",
            "SAMARIUM": "62", "Sm": "62", "62": "62", "SM": "62",
            "EUROPIUM": "63", "Eu": "63", "63": "63", "EU": "63",
            "GADOLINIUM": "64", "Gd": "64", "64": "64", "GD": "64",
            "TERBIUM": "65", "Tb": "65", "65": "65", "TB": "65",
            "DYSPROSIUM": "66", "Dy": "66", "66": "66", "DY": "66",
            "HOLMIUM": "67", "Ho": "67", "67": "67", "HO": "67",
            "ERBIUM": "68", "Er": "68", "68": "68", "ER": "68",
            "THULIUM": "69", "TM": "69", "69": "69",
            "YTTERBIUM": "70", "Yb": "70", "70": "70", "YB": "70",
            "LUTETIUM": "71", "Lu": "71", "71": "71", "LU": "71",
            "HAFNIUM": "72", "Hf": "72", "72": "72", "HF": "72",
            "TANTALUM": "73", "Ta": "73", "73": "73", "TA": "73",
            "TUNGSTEN": "74", "W": "74", "74": "74",
            "RHENIUM": "75", "Re": "75", "75": "75", "RE": "75",
            "OSMIUM": "76", "Os": "76", "76": "76", "OS": "76",
            "IRIDIUM": "77", "Ir": "77", "77": "77", "IR": "77",
            "PLATINUM": "78", "Pt": "78", "78": "78", "PT": "78",
            "GOLD": "79", "Au": "79", "79": "79", "AU": "79",
            "MERCURY": "80", "Hg": "80", "80": "80", "HG": "80",
            "THALLIUM": "81", "Tl": "81", "81": "81", "TL": "81",
            "LEAD": "82", "Pb": "82", "82": "82", "PB": "82",
            "BISMUTH": "83", "Bi": "83", "83": "83", "BI": "83",
            "POLONIUM": "84", "Po": "84", "84": "84", "PO": "84",
            "ASTATINE": "85", "At": "85", "85": "85", "AT": "85",
            "RADON": "86", "Rn": "86", "86": "86", "RN": "86"}

        fullname = ["HYDROGEN", "HELIUM", "LITHIUM", "BERYLLIUM", "BORON", "CARBON", "NITROGEN", "OXYGEN", "FLUORINE",
                    "NEON",
                    "SODIUM", "MAGNESIUM", "ALUMINUM", "SILICON", "PHOSPHORUS", "SULFUR", "CHLORINE", "ARGON",
                    "POTASSIUM", "CALCIUM",
                    "SCANDIUM", "TITANIUM", "VANADIUM", "CHROMIUM", "MANGANESE", "IRON", "COBALT", "NICKEL", "COPPER",
                    "ZINC",
                    "GALLIUM", "GERMANIUM", "ARSENIC", "SELENIUM", "BROMINE", "KRYPTON", "RUBIDIUM", "STRONTIUM",
                    "YTTRIUM", "ZIRCONIUM",
                    "NIOBIUM", "MOLYBDENUM", "TECHNETIUM", "RUTHENIUM", "RHODIUM", "PALLADIUM", "SILVER", "CADMIUM",
                    "INDIUM", "TIN",
                    "ANTIMONY", "TELLURIUM", "IODINE", "XENON", "CESIUM", "BARIUM", "LANTHANUM", "CERIUM",
                    "PRASEODYMIUM", "NEODYMIUM",
                    "PROMETHIUM", "SAMARIUM", "EUROPIUM", "GADOLINIUM", "TERBIUM", "DYSPROSIUM", "HOLMIUM", "ERBIUM",
                    "THULIUM", "YTTERBIUM",
                    "LUTETIUM", "HAFNIUM", "TANTALUM", "TUNGSTEN", "RHENIUM", "OSMIUM", "IRIDIUM", "PLATINUM", "GOLD",
                    "MERCURY",
                    "THALLIUM", "LEAD", "BISMUTH", "POLONIUM", "ASTATINE", "RADON"]

        symbol = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
                  "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
                  "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
                  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "TM", "Yb",
                  "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                  "Tl", "Pb", "Bi", "Po", "At", "Rn"]

        mass = [1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.180,
                22.990, 24.305, 26.982, 28.086, 30.974, 32.065, 35.453, 39.948, 39.098, 40.078,
                44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.390,
                69.723, 72.640, 74.922, 78.960, 79.904, 83.800, 85.468, 87.620, 88.906, 91.224,
                92.906, 95.940, 98.000, 101.070, 102.906, 106.420, 107.868, 112.411, 114.818, 118.710,
                121.760, 127.600, 126.905, 131.293, 132.906, 137.327, 138.906, 140.116, 140.908, 144.240,
                145.000, 150.360, 151.964, 157.250, 158.925, 162.500, 164.930, 167.259, 168.934, 173.040,
                174.967, 178.490, 180.948, 183.840, 186.207, 190.230, 192.217, 195.078, 196.967, 200.590,
                204.383, 207.200, 208.980, 209.000, 210.000, 222.000]

        # Van der Waals Radius, missing data replaced by 2.00
        radii = [1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,
                 2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31,
                 2.11, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.40, 1.39,
                 1.87, 2.11, 1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 2.00, 2.00,
                 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.72, 1.58, 1.93, 2.17,
                 2.00, 2.06, 1.98, 2.16, 3.43, 2.68, 2.00, 2.00, 2.00, 2.00,
                 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
                 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.75, 1.66, 1.55,
                 1.96, 2.02, 2.07, 1.97, 2.02, 2.20]

        self.__name = int(periodic_table[name])
        self.__FullName = fullname[self.__name - 1]
        self.__Symbol = symbol[self.__name - 1]
        self.__Mass = mass[self.__name - 1]
        self.__Radii = radii[self.__name - 1]

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


def BND(xyz, var):
    ## This function calculate distance
    ## a<->b

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b = var[0:2]

    v1 = xyz[a - 1]
    v2 = xyz[b - 1]
    r = la.norm(v1 - v2)
    return r


def AGL(xyz, var):
    ## This function calculate angle
    ## a<-b->c

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c = var[0:3]

    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v1 = v1 / la.norm(v1)
    v2 = v2 / la.norm(v2)
    cosa = np.dot(v1, v2)
    alpha = np.arccos(cosa) * 57.2958
    return alpha


def DHD(xyz, var):
    ## This function calculate dihedral angle
    ##   n1    n2
    ##    |    |
    ## a<-b-><-c->d

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d = var[0:4]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v3 = r2 - r3
    v4 = r4 - r3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958

    return beta


def DHD2(xyz, var):
    ## This function calculate dihedral angle
    ##   n1    n2
    ##    |    |
    ## a<-b-><-c->d

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d = var[0:4]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v3 = r2 - r3
    v4 = r4 - r3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958
    axis = np.cross(n1, n2)
    pick = np.argmax(np.abs(axis))
    # find the projection with the largest magnitude (non-zero), then just compare it to avoid 0/0
    sign = np.sign(axis[pick] / v2[pick])
    if sign == -1:
        beta = - beta

    return beta


def DHD3(xyz, var):
    ## This function calculate dihedral angle
    ##   n1    n2
    ##    |    |
    ## a<-b-><-c->d

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d = var[0:4]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v3 = r2 - r3
    v4 = r4 - r3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958
    axis = np.cross(n1, n2)
    pick = np.argmax(np.abs(axis))
    # find the projection with the largest magnitude (non-zero), then just compare it to avoid 0/0
    sign = np.sign(axis[pick] / v2[pick])
    if sign == -1:
        beta = 360 - beta

    return beta


def DHDD(xyz, var):
    ## This function calculate dihedral angle involving dummy center
    ##   n1    n2
    ##    |    |
    ## a,b<-c-><-d->e,f

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d, e, f = var[0:6]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    r5 = np.array(xyz[e - 1])
    r6 = np.array(xyz[f - 1])
    v1 = (r1 + r2) / 2 - r3
    v2 = r4 - r3
    v3 = r3 - r4
    v4 = (r5 + r6) / 2 - r4
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958
    axis = np.cross(n1, n2)
    pick = np.argmax(np.abs(axis))
    # find the projection with the largest magnitude (non-zero), then just compare it to avoid 0/0
    sign = np.sign(axis[pick] / v2[pick])
    if sign == -1:
        beta = 360 - beta

    return beta

def DHDD2(xyz, var):
    ## This function calculate dihedral angle involving dummy center
    ##   n1    n2
    ##    |    |
    ## a,b<-c-><-d->e,f

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d, e, f = var[0:6]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    r5 = np.array(xyz[e - 1])
    r6 = np.array(xyz[f - 1])
    v1 = (r1 + r2) / 2 - r3
    v2 = r4 - r3
    v3 = r3 - r4
    v4 = (r5 + r6) / 2 - r4
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958
    axis = np.cross(n1, n2)
    pick = np.argmax(np.abs(axis))
    # find the projection with the largest magnitude (non-zero), then just compare it to avoid 0/0
    sign = np.sign(axis[pick] / v2[pick])
    if sign == -1:
        beta = - beta

    return beta

def DHDD3(xyz, var):
    ## This function calculate dihedral angle involving dummy center
    ##   n1    n2
    ##    |    |
    ## a,b<-c-><-d->e,f

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d, e, f = var[0:6]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    r5 = np.array(xyz[e - 1])
    r6 = np.array(xyz[f - 1])
    v1 = (r1 + r2) / 2 - r3
    v2 = r4 - r3
    v3 = r3 - r4
    v4 = (r5 + r6) / 2 - r4
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    beta = np.arccos(cosb) * 57.2958
    axis = np.cross(n1, n2)
    pick = np.argmax(np.abs(axis))
    # find the projection with the largest magnitude (non-zero), then just compare it to avoid 0/0
    sign = np.sign(axis[pick] / v2[pick])
    if sign == -1:
        beta = 360 - beta

    return beta

def OOP(xyz, var):
    ## This function calculate out-of-plane angle
    ##    n  d
    ##    |  |
    ## a<-b->c

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d = var[0:4]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v3 = r4 - r3
    v3 = v3 / la.norm(v3)
    n = np.cross(v1, v2)
    n = n / la.norm(n)
    cosb = np.dot(n, v3)
    gamma = np.arccos(cosb) * 57.2958

    return gamma


def PPA(xyz, var):
    ## This function calculate plane-plane angle
    ##   n1       n2
    ##    |       |
    ## a<-b->c d<-e->f

    var = [int(x) for x in var]
    xyz = np.array([[float(y) for y in x.split()[1: 4]] for x in xyz])
    a, b, c, d, e, f = var[0:6]
    r1 = np.array(xyz[a - 1])
    r2 = np.array(xyz[b - 1])
    r3 = np.array(xyz[c - 1])
    r4 = np.array(xyz[d - 1])
    r5 = np.array(xyz[e - 1])
    r6 = np.array(xyz[f - 1])
    v1 = r1 - r2
    v2 = r3 - r2
    v3 = r4 - r5
    v4 = r6 - r5
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v4)
    n1 = n1 / la.norm(n1)
    n2 = n2 / la.norm(n2)
    cosb = np.dot(n1, n2)
    delta = np.arccos(cosb) * 57.2958
    return delta


def G(coord, par_sym):
    ## This function calculate symmetry function in Behler, J, Int. J. Quantum Chem., 2-15, 115 1032-1050
    ## This function return a list of values for each atom
    ## coord is a numpy array of floating numbers
    ## par_sym has default values in RMSD

    cut = par_sym['cut']  # cutoff function version 1 or 2
    ver = par_sym['ver']  # symmetry function 1-4
    rc = par_sym['rc']  # cutoff radii, 0 is the maximum * 1.1
    n = par_sym['n']  # Gaussian exponent
    rs = par_sym['rs']  # Gaussian center
    z = par_sym['z']  # angular exponent
    ll = par_sym['l']  # cosine exponent, only 1 or -1

    # print('\nSymmetry function: %d' % (ver))
    # print('Cutoff function: %d' % (cut))
    # print('Cutoff radii:%6.2f Shift:%6.2f' % (rc,rs))
    # print('Eta:%6.2f Zeta:%6.2f Lambda:%6.2f' % (n,z,l))

    ## prepare distance matrix
    dist = np.array([[0.0 for _ in coord] for _ in coord])
    for n, i in enumerate(coord):
        for m, j in enumerate(coord):
            if n != m:  # update dist if n != m
                r = la.norm(i - j)
                dist[n][m] = r

    ## prepare cutoff function matrix
    if rc == 0:
        rc = 1.1 * np.amax(dist)
    fc = np.array([[1.0 for _ in coord] for _ in coord])
    for n, i in enumerate(dist):
        for m, j in enumerate(dist):
            r = dist[n][m]
            r /= rc
            if r < 1:
                fc[n][m] = r  # update fc if r < rc

    ## prepare angle matrix if needed, i is the center atom!
    angl = np.zeros(0)
    if ver > 2:
        angl = np.array([[[0.0 for _ in coord] for _ in coord] for _ in coord])
        for n, i in enumerate(coord):
            for m, j in enumerate(coord):
                for o, k in enumerate(coord):
                    if n != m and m != o and o != n:
                        v1 = j - i
                        v2 = j - k
                        v1 = v1 / la.norm(v1)
                        v2 = v2 / la.norm(v2)
                        cosa = np.dot(v1, v2)
                        alpha = np.arccos(cosa) * 57.2958
                        angl[n][m][o] = alpha
    if cut == 1:
        fc = 0.5 * np.cos(np.pi * fc) + 1
    elif cut == 2:
        fc = np.tanh(1 - fc) ** 3
    else:
        print('\n!!! Cannot recognize cutoff function !!!\n')
        exit()

    if ver == 1:
        g = np.sum(fc, axis=1)
    elif ver == 2:
        w = np.exp((-1) * n * (dist - rs) ** 2)
        g = np.sum(w * fc, axis=1)
    elif ver == 3:
        g = np.array([0.0 for _ in coord])
        for i in range(len(coord)):
            for j in range(len(coord)):
                for k in range(len(coord)):
                    a = (1 + ll * np.cos(angl[i][j][k])) ** z
                    w = np.exp((-1) * n * (dist[i][j] ** 2 + dist[i][k] ** 2 + dist[j][k] ** 2))
                    f = fc[i][j] * fc[i][k] * fc[j][k]
                    g[i] += 2 ** (1 - z) * a * w * f
    elif ver == 4:
        g = np.array([0.0 for _ in coord])
        for i in range(len(coord)):
            for j in range(len(coord)):
                for k in range(len(coord)):
                    a = (1 + ll * np.cos(angl[i][j][k])) ** z
                    w = np.exp((-1) * n * (dist[i][j] ** 2 + dist[i][k] ** 2))
                    f = fc[i][j] * fc[i][k]
                    g[i] += 2 ** (1 - z) * a * w * f
    else:
        g = np.zeros(0)
        print('\n!!! Cannot recognize symmetry function !!!\n')
        exit()
    return g


def RMSD(xyz, ref, var):
    ## This function calculate RMSD between product and reference
    ## This function call kabsch to reduce RMSD between product and reference
    ## This function call hungarian to align product and reference

    ## general variables for all functions
    excl = []  # exclude elements
    incl = []  # only include elements
    pck = []  # pick this atoms
    align = 'NO'  # do not align product and reference
    coord = 'CART'  # use cartesian coordinates
    rmsd = 'NONE'  # rmsd have not been calculated

    ## symmetry function default variables
    par_sym = {
        'cut': 1,  # cutoff function version 1 or 2
        'ver': 1,  # symmetry function 1-4
        'rc': 6,  # cutoff radii, 0 is the maximum * 1.1
        'n': 1.2,  # Gaussian exponent
        'rs': 0,  # Gaussian center
        'z': 1,  # angular exponent
        'l': 1  # cosine factor, only 1 or -1
    }

    for i in var:
        i = i.upper()
        if 'NO=' in i:
            e = i.split('=')[1]
            e = Element(e).getSymbol()
            excl.append(e)
        elif 'ON=' in i:
            e = i.split('=')[1]
            e = Element(e).getSymbol()
            incl.append(e)
        elif 'PICK=' in i:
            pck = [int(x) for x in i.split('=')[1].split(',')]
        elif 'ALIGN=' in i:
            i = i.split('=')[1]
            if i == 'HUNG' or i == 'NO':  # align must be either hung or no
                align = i
        elif 'COORD=' in i:
            i = i.split('=')[1]
            if i == 'CART' or i == 'SYM':  # coord must be either cart or sym
                coord = i
        elif 'CUT=' in i:
            i = int(i.split('=')[1])
            if i in [1, 2]:  # cut must be within 1-2
                par_sym['cut'] = i
        elif 'VER=' in i:
            i = int(i.split('=')[1])
            if i in [1, 2, 3, 4]:  # ver must be within 1-4
                par_sym['ver'] = i
        elif 'RC=' in i:
            par_sym['rc'] = float(i.split('=')[1])
        elif 'ETA=' in i:
            par_sym['n'] = float(i.split('=')[1])
        elif 'RS=' in i:
            par_sym['rs'] = float(i.split('=')[1])
        elif 'ZETA=' in i:
            par_sym['z'] = float(i.split('=')[1])
        elif 'LAMBDA=' in i:
            ll = int(i.split('=')[1])
            if ll >= 0:  # l only takes +1 or -1
                par_sym['l'] = 1
            else:
                par_sym['l'] = -1

    ## prepare atom list and coordinates
    el = []  # element list
    for i in xyz:
        e, x, y, z = i.split()
        e = Element(e).getSymbol()
        if e not in el:
            el.append(e)

    if len(excl) > 0:
        el = [x for x in el if x not in excl]
    if len(incl) > 0:
        el = [x for x in el if x in incl]

    s = [x + 1 for x in range(len(xyz))]  # atom index list
    if len(pck) > 0:
        s = pck

    p = []  # product coordinates
    patoms = []
    for n, i in enumerate(xyz):
        e, x, y, z = i.split()
        x, y, z = float(x), float(y), float(z)
        e = Element(e).getSymbol()
        if e in el and n + 1 in s:
            p.append([x, y, z])
            patoms.append(e)
    p = np.array(p)

    q = []  # reference coordinates
    qatoms = []
    for n, i in enumerate(ref):
        e, x, y, z = i.split()
        x, y, z = float(x), float(y), float(z)
        e = Element(e).getSymbol()
        if e in el and n + 1 in s:
            q.append([x, y, z])
            qatoms.append(e)
    q = np.array(q)

    p -= p.mean(axis=0)  # translate to the centroid
    q -= q.mean(axis=0)  # translate to the centroid

    if align == 'HUNG':  # align coordinates
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

        order = []
        rmsd = []
        for sw in swap:
            for r in reflection:
                tatoms = [x for x in qatoms]
                t = np.array([x for x in q])
                t = t[:, sw]
                t = np.dot(t, np.diag(r))
                t -= t.mean(axis=0)
                ip = inertia(patoms, p)
                it = inertia(tatoms, t)
                u1 = rotate(ip, it)
                u2 = rotate(ip, -it)
                t1 = np.dot(t, u1)
                t2 = np.dot(t, u2)
                order1 = hungarian(patoms, tatoms, p, t1)
                order2 = hungarian(patoms, tatoms, p, t2)
                rmsd1 = kabsch(p, t[order1])
                rmsd2 = kabsch(p, t[order2])
                order += [order1, order2]
                rmsd += [rmsd1, rmsd2]
        pick = np.argmin(rmsd)
        order = order[pick]
        rmsd = rmsd[pick]
        q = q[order]
    if coord == 'SYM':  # use symmetry function
        g_prd = G(p, par_sym)
        g_ref = G(q, par_sym)
        rmsd = np.sqrt(np.sum((g_prd - g_ref) ** 2) / len(g_prd))

    if rmsd == 'NONE':
        rmsd = kabsch(p, q)

    return rmsd


def kabsch(p, q):
    ## This function use Kabsch algorithm to reduce RMSD by rotation

    c = np.dot(np.transpose(p), q)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:  # ensure right-hand system
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    p = np.dot(p, u)
    diff = p - q
    n = len(p)
    return np.sqrt((diff * diff).sum() / n)


def inertia(atoms, xyz):
    ## This function calculate principal axis

    xyz = np.array([i for i in xyz])  # copy the array to avoid changing it
    mass = []
    for i in atoms:
        m = Element(i).getMass()
        mass.append(m)
    mass = np.array(mass)
    xyz -= np.average(xyz, weights=mass, axis=0)
    xx = 0.0
    yy = 0.0
    zz = 0.0
    xy = 0.0
    xz = 0.0
    yz = 0.0
    for n, i in enumerate(xyz):
        xx += mass[n] * (i[1] ** 2 + i[2] ** 2)
        yy += mass[n] * (i[0] ** 2 + i[2] ** 2)
        zz += mass[n] * (i[0] ** 2 + i[1] ** 2)
        xy += -mass[n] * i[0] * i[1]
        xz += -mass[n] * i[0] * i[2]
        yz += -mass[n] * i[1] * i[2]

    im = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    eigval, eigvec = np.linalg.eig(im)

    return eigvec[np.argmax(eigval)]


def rotate(p, q):
    ## This function calculate the matrix rotate q onto p
    p: np.ndarray
    q: np.ndarray

    if (p == q).all():
        return np.eye(3)
    elif (p == -q).all():
        # return a rotation of pi around the y-axis
        return np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    else:
        v = np.cross(p, q)
        s = np.linalg.norm(v)
        c = np.vdot(p, q)
        vx = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        return np.eye(3) + vx + np.dot(vx, vx) * ((1. - c) / (s * s))


def hungarian(patoms, qatoms, p, q):
    ## This function use hungarian algorithm to align P onto Q
    ## This function call linear_sum_assignment from scipy to solve hungarian problem
    ## This function call inertia to find principal axis
    ## This function call rotate to rotate P onto aligned Q

    unique_atoms = np.unique(patoms)

    reorder = np.zeros(len(qatoms), dtype=int)
    for atom in unique_atoms:
        pidx = []
        qidx = []

        for n, p in enumerate(patoms):
            if p == atom:
                pidx.append(n)
        for m, q in enumerate(qatoms):
            if q == atom:
                qidx.append(m)

        pidx = np.array(pidx)
        qidx = np.array(qidx)
        a = p[pidx]
        b = q[qidx]
        ab = np.array([[la.norm(aa - bb) for bb in b] for aa in a])
        aidx, bidx = linear_sum_assignment(ab)
        reorder[pidx] = qidx[bidx]
    return reorder


def getindex(index):
    ## This function read single, range, separate range index and convert them to a list
    index_list = []
    for i in index:
        if '-' in i:
            a, b = i.split('-')
            a, b = int(a), int(b)
            index_list += range(a, b + 1)
        else:
            index_list.append(int(i))

    index_list = sorted(list(set(index_list)))  # remove duplicates and sort from low to high
    return index_list


def redindex(index):
    ## This function compress a list of index into range
    index = sorted(list(set(index)))
    groups = []
    subrange = []
    for i in index:
        subrange.append(int(i))
        if len(subrange) > 1:
            d = subrange[-1] - subrange[-2]  # check continuity
            if d > 1:
                groups.append(subrange[0:-1])
                subrange = [subrange[-1]]
        if i == index[-1]:
            groups.append(subrange)

    index_range = ''
    for j in groups:
        if len(j) == 1:
            index_range += '%s ' % (j[0])
        elif len(j) == 2:
            index_range += '%s %s ' % (j[0], j[1])
        else:
            index_range += '%s-%s ' % (j[0], j[-1])
    return index_range


def set_prune(prune_type, prune_index, prune_thrhd):
    prune_index = ' '.join(prune_index).split(',')
    prune_index = [x.split() for x in prune_index]
    pindex = []
    pthrhd = []
    if 'frag' in prune_type:
        p1 = np.array(getindex(prune_index[0])) - 1
        p2 = np.array(getindex(prune_index[1])) - 1
        p3 = prune_thrhd[0]
        pindex.append([p1, p2])
        pthrhd.append(p3)
    else:
        diff = len(prune_index) - len(prune_thrhd)
        if diff > 0:
            add = [prune_thrhd[-1] for _ in range(diff)]
            prune_thrhd = prune_thrhd + add
        else:
            prune_thrhd = prune_thrhd[:len(prune_index)]

        for n, p in enumerate(prune_index):
            p1, p2 = getindex(p)[0: 2]
            p3 = prune_thrhd[n]
            pindex.append([[p1 - 1], [p2 - 1]])
            pthrhd.append(p3)

    return pindex, pthrhd


def check_param(coord, src, dst, thrhd):
    coord = np.array([x.split()[1: 4] for x in coord]).astype(float)
    a = np.mean(coord[src], axis=0)
    b = np.mean(coord[dst], axis=0)
    d = np.sum((a - b) ** 2) ** 0.5

    if d > thrhd:
        return True, d

    return False, d


def format1(n, xyz):
    ## This function convert coordinates list to string
    output = '%d\n' % n
    for i in xyz:
        output += '%s\n' % i
    return output


def format2(x):
    ## This function convert a one-line string to multiple lines:
    str_new = ''
    for n, i in enumerate(x.split()):
        str_new += '%10s ' % i
        if (n + 1) % 10 == 0:
            str_new += '\n'
        else:
            if i == x.split()[-1]:
                str_new += '\n'
    return str_new


def format3(n, xyz):
    ## This function convert coordinates from a list of string to float
    output = []
    for line in xyz[1: 1 + n]:
        atom, x, y, z = line.split()[0: 4]
        output.append([str(atom), float(x), float(y), float(z)])

    return output


def format4(xyz):
    ## This function convert coordinates from a list of float to string
    output = []
    for line in xyz:
        atom, x, y, z = line
        output.append('%-5s%16.8f%16.8f%16.8f' % (atom, x, y, z))

    return output


def Refread(ref):
    ## This function read the reference structure from a file: ref
    ref_coord = []
    with open(ref, 'r') as refxyz:
        coord = refxyz.read().splitlines()
    natom = int(coord[0])
    n = 0
    m = 0
    for _ in coord:
        n += 1
        if n % (natom + 2) == 0:  # at the last line of each coordinates
            m += 1
            ref_coord.append(coord[n - natom:n])
    print('\nRead reference structures: %5d in %s\n' % (m, ref))
    return ref_coord


def Paramread(param):
    ## This function read the geometrical parameters from string or a file

    p_list = ['B', 'A', 'D', 'D2', 'D3', 'DD', 'DD2', 'DD3', 'O', 'P', 'RMSD']
    parameters = []
    par_group = []

    for n, p in enumerate(param):
        if p in p_list:
            if len(par_group) > 0:
                parameters.append(par_group)
            par_group = [p]
        else:
            par_group.append(p)

        if n == len(param) - 1:
            parameters.append(par_group)

    return parameters


def Pararead(parameters):
    ## This function read geometrical parameters and return the name and comment of parameters
    name = []
    cmmt = []
    for i in parameters:
        i = [x.upper() for x in i]
        name.append(i[0])
        if i[0] != 'RMSD':
            cmmt.append(','.join(i[1:]))
        else:
            v1 = 'D'
            v2 = ' CART'
            if 'ALIGN=HUNG' in i:
                v1 = 'A'
            if 'COORD=SYM' in i:
                v2 = '  SYM'
            cmmt.append(v1 + v2)
    return name, cmmt


def compute_para(var):
    ## This function wrap the evaluate for parallelization
    i_traj, i_geom, i_param, geom, param, thrhd, ref_coord = var
    geom_param = evaluate(geom, ref_coord, param)
    geom_type = int(geom_param > thrhd)

    return i_traj, i_geom, i_param, geom_param, geom_type


def evaluate(coord, ref_coord, param):
    ## This function evaluate the value of the parameter given a coordinate
    func = param[0]
    var = param[1:]

    if func == 'B':
        method = BND
    elif func == 'A':
        method = AGL
    elif func == 'D':
        method = DHD
    elif func == 'D2':
        method = DHD2
    elif func == 'D3':
        method = DHD3
    elif func == 'DD':
        method = DHDD
    elif func == 'DD2':
        method = DHDD2
    elif func == 'DD3':
        method = DHDD3
    elif func == 'O':
        method = OOP
    elif func == 'P':
        method = PPA
    elif func == 'RMSD':
        method = RMSD
    else:
        method = None
        print('Method is not found: %s' % func)
        exit()

    if func == 'RMSD':
        geom_param = method(coord, ref_coord, var)
    else:
        geom_param = method(coord, var)

    return geom_param


def count_data_lines_molcas(files):
    ## This function count lines for molcas data from provided folders
    ## This function is for parallelization
    ntraj, f = files
    t = f.split('/')[-1]
    if os.path.exists('%s/%s.log' % (f, t)):
        with open('%s/%s.log' % (f, t), 'r') as logfile:
            log = logfile.read().splitlines()
        n = 0
        for line in log:
            if 'Gnuplot' in line:
                n += 1
        nlog = n
    elif os.path.exists('%s/%s.out' % (f, t)):
        with open('%s/%s.out' % (f, t), 'r') as logfile:
            log = logfile.read().splitlines()
        n = 0
        for line in log:
            if 'Gnuplot' in line:
                n += 1
        nlog = n
    else:
        nlog = 0

    if os.path.exists('%s/%s.md.energies' % (f, t)):
        with open('%s/%s.md.energies' % (f, t), 'r') as engfile:
            eng = engfile.read().splitlines()
        nenergy = len(eng) - 1
    else:
        nenergy = 0

    if os.path.exists('%s/%s.md.xyz' % (f, t)):
        with open('%s/%s.md.xyz' % (f, t), 'r') as xyzfile:
            xyz = xyzfile.read().splitlines()
        natom = int(xyz[0])
        nxyz = len(xyz) / (natom + 2)
    else:
        nxyz = 0

    return ntraj, f, nlog, nenergy, nxyz


def count_data_lines_newtonx(files):
    ## This function count lines for newtonx data from provided folders
    ## This function is for parallelization
    ntraj, f = files
    if os.path.exists('%s/RESULTS/dyn.out' % f):
        with open('%s/RESULTS/dyn.out' % f, 'r') as logfile:
            log = logfile.read().splitlines()
        n = 0
        for line in log:
            if 'STEP' in line:
                n += 1
        nlog = n - 1
        nxyz = n - 1
    else:
        nlog = 0
        nxyz = 0

    if os.path.exists('%s/RESULTS/en.dat' % f):
        with open('%s/RESULTS/en.dat' % f, 'r') as engfile:
            eng = engfile.read().splitlines()
        nenergy = len(eng)
    else:
        nenergy = 0

    return ntraj, f, nlog, nenergy, nxyz


def count_data_lines_sharc(files):
    ## This function count lines for sharc data from provided folders
    ## This function is for parallelization
    ntraj, f = files
    if os.path.exists('%s/output.dat' % f):
        with open('%s/output.dat' % f, 'r') as logfile:
            log = logfile.read().splitlines()
        n = 0
        for line in log:
            if '! 0 Step' in line:
                n += 1
        nlog = n - 1
        nenergy = n - 1
        nxyz = n - 1
    else:
        nlog = 0
        nenergy = 0
        nxyz = 0

    return ntraj, f, nlog, nenergy, nxyz


def count_data_lines_fromage(files):
    ## This function count lines for fromage data from provided folders
    ## This function is for parallelization
    ntraj, f = files
    if os.path.exists('%s/output.chk' % f):
        with open('%s/output.chk' % f, 'r') as logfile:
            log = logfile.read().splitlines()
        n = len(log) - 3
        nlog = n
        nenergy = n
        nxyz = n
    else:
        nlog = 0
        nenergy = 0
        nxyz = 0

    return ntraj, f, nlog, nenergy, nxyz


def read_raw_data_molcas(files):
    ## This function read data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and
    ## append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj, f, maxstep, pindex, pthrhd = files
    t = f.split('/')[-1]

    if os.path.exists('%s/%s.log' % (f, t)):
        with open('%s/%s.log' % (f, t), 'r') as logfile:
            log = logfile.read().splitlines()

    elif os.path.exists('%s/%s.out' % (f, t)):
        with open('%s/%s.out' % (f, t), 'r') as logfile:
            log = logfile.read().splitlines()

    with open('%s/%s.md.energies' % (f, t), 'r') as engfile:
        eng = engfile.read().splitlines()

    with open('%s/%s.md.xyz' % (f, t), 'r') as xyzfile:
        xyz = xyzfile.read().splitlines()

    ## Find population and label for each structure
    trj_pop = []  # population list
    trj_lab = []  # label list
    trj_hop = []  # hopping event list
    crt_label = ''  # current label
    crt_hop = 0  # current index for hopping event
    crt_state = 0  # current state
    ini_state = 0  # initial state
    pmd = 0  # flag to read pyrai2md log
    n = 0  # count line number
    i = 0  # count structure number
    version = 19
    for line in log:
        n += 1
        if 'version' in line:
            try:
                version = float(line.split()[-1])
            except ValueError:
                version = 18
            except IndexError:
                version = 18

        if 'PyRAI2MD' in line:
            pmd = 1

        if 'Root chosen for geometry opt' in line:
            i += 1
            crt_state = int(line.split()[-1]) - 1
            if i == 1:
                ini_state = crt_state
            crt_label = 'traj %d coord %d init %s state %s' % (ntraj + 1, i, ini_state, crt_state)

        if 'Gnuplot' in line:
            # it is more reliable to count crt_hop separately, although it equals to i if the job complete normally.
            crt_hop += 1
            if version > 18 or pmd == 1:
                event_checker = log[n + 3]
                event_info = log[n + 5]
                num_state = int((len(line.split()) - 2) / 2)
                crt_pop = line.split()[1:num_state + 1]
                crt_pop = [float(i) for i in crt_pop]
            else:
                if np.abs(float(line.split()[-1])) < 2:
                    # num_state = len(line.split()) - 1
                    event_checker = log[n + 4]
                    event_info = log[n + 6]
                else:
                    # num_state = int((len(line.split()) - 2) / 2)
                    event_checker = log[n + 3]
                    event_info = log[n + 5]
                crt_pop = line.split()[1:]
                crt_pop = [float(i) for i in crt_pop]

            if 'event' in event_checker:
                hop_state = int(event_info.split()[-2])
                crt_label += ' to %d CI' % (hop_state - 1)
                trj_hop.append(crt_hop)
                crt_state = hop_state - 1

            trj_lab.append(crt_label)
            trj_pop.append(crt_pop)
            if i == maxstep and maxstep != 0:  # cutoff trajectories, mstep is the global variable of cutoff step
                break

    nstate = len(trj_pop[0])
    ## Find kinetic and potential energy for each state and time step
    trj_kin = []  # kinetic energy list
    trj_pot = []  # potential energy list
    trj_tim = []  # time step list
    n = 0
    for line in eng:
        n += 1
        if 'time' in line:
            continue  # skip the title line
        line = line.replace('D', 'e')
        line = line.split()
        crt_time = float(line[0])
        crt_kin = float(line[2])
        crt_pot = line[4:nstate + 4]
        crt_pot = [float(i) for i in crt_pot]
        trj_kin.append(crt_kin)
        trj_pot.append(crt_pot)
        trj_tim.append(crt_time)
        if n - 1 == maxstep and maxstep != 0:  # cutoff trajectories
            break

    dtime = trj_tim[1] - trj_tim[0]
    ## Find coordinates
    trj_coord = []  # coordinates list
    natom = int(xyz[0])
    n = 0  # count line number
    m = 0  # count structure number
    pstep = 0  # count step to prune trajectory
    for _ in xyz:
        n += 1
        if n % (natom + 2) == 0:  # at the last line of each coordinates
            m += 1
            if len(pindex) > 0 and pstep == 0:
                for k, p in enumerate(pindex):
                    p1, p2 = p
                    p3 = pthrhd[k]
                    stop, d = check_param(xyz[n - natom:n], p1, p2, p3)
                    if stop:
                        pstep = m
                        break

            crt_coord = [trj_lab[m - 1]] + xyz[n - natom:n]  # combine label with coordinates
            trj_coord.append(crt_coord)

            if m == maxstep and maxstep != 0:  # cutoff trajectories
                break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab = len(trj_lab)
    len_pop = len(trj_pop)
    len_kin = len(trj_kin)
    len_pot = len(trj_pot)
    len_crd = len(trj_coord)
    nstep = np.amin([len_lab, len_pop, len_kin, len_pot, len_crd]).tolist()

    trj_lab = trj_lab[0:nstep]
    trj_pop = trj_pop[0:nstep]
    trj_kin = trj_kin[0:nstep]
    trj_pot = trj_pot[0:nstep]
    trj_coord = trj_coord[0:nstep]

    if pstep > 0:
        trj_lab = trj_lab[0:pstep]
        trj_pop = trj_pop[0:pstep]
        trj_kin = trj_kin[0:pstep]
        trj_pot = trj_pot[0:pstep]
        trj_coord = trj_coord[0:pstep]
        nstep = pstep
        crt_state = int(trj_lab[-1].split()[7])
        p_hop = trj_hop
        trj_hop = []
        for s in p_hop:
            if s <= pstep:
                trj_hop.append(s)

    return ntraj, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state


def read_raw_data_newtonx(files):
    ## This function read newtonx data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and
    ## append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj, f, maxstep, pindex, pthrhd = files

    with open('%s/RESULTS/nx.log' % f, 'r') as nxfile:
        nx = nxfile.read().splitlines()

    with open('%s/RESULTS/dyn.out' % f, 'r') as logfile:
        log = logfile.read().splitlines()

    with open('%s/RESULTS/sh.out' % f, 'r') as popfile:
        pop = popfile.read().splitlines()

    with open('%s/RESULTS/en.dat' % f, 'r') as engfile:
        eng = engfile.read().splitlines()

    ## Find label and coordinates for each structure
    natom = 0  # number of atoms
    nstate = 0  # number of states
    trj_lab = []  # label list
    trj_hop = []  # hopping event list
    trj_coord = []  # coordinates list
    crt_state = 0  # current state
    ini_state = 0  # initial state
    pre_state = 0  # previous state
    n = 0  # count line number
    i = 0  # count structure number
    pstep = 0  # count step to prune trajectory

    for line in nx:
        if 'Nat' in line:
            natom = int(line.split()[-1])
        if 'nstat ' in line:
            nstate = int(line.split()[-1])
        if 'etot_drift' in line:
            break

    for line in log:
        n += 1
        if 'Molecular dynamics on state' in line:
            i += 1
            crt_state = int(line.split()[6]) - 1
            if i == 1:
                ini_state = crt_state
                pre_state = crt_state

            crt_label = 'traj %d coord %d init %s' % (ntraj + 1, i, ini_state)

            if pre_state != crt_state:
                crt_hop = i
                hop_state = crt_state
                crt_label += ' state %d to %d CI' % (pre_state, hop_state)
                trj_hop.append(crt_hop)
                pre_state = crt_state
            else:
                crt_label += ' state %d' % crt_state

            xyz = log[n - 1 + 4:n - 1 + natom + 4]  # find coordinates
            xyz = ['%-5s %14.8f %14.8f %14.8f' % (
                x.split()[0], float(x.split()[2]) * 0.529177, float(x.split()[3]) * 0.529177,
                float(x.split()[4]) * 0.529177) for x in xyz]  # pick atom,x,y,z
            crt_coord = [crt_label] + xyz  # combine label with coordinates

            if pindex and pstep == 0:
                for k, p in enumerate(pindex):
                    p1, p2 = p
                    p3 = pthrhd[k]
                    stop, d = check_param(xyz, p1, p2, p3)
                    if stop:
                        pstep = i
                        break

            trj_lab.append(crt_label)
            trj_coord.append(crt_coord)

            if i == maxstep and maxstep != 0:  # cutoff trajectories
                break
    trj_lab = trj_lab[:-1]
    trj_coord = trj_coord[:-1]  # newtonx does not compute the last geometry

    ## Find state population for each structure
    trj_pop = []  # population list
    n = 0  # count line number
    i = 0  # count structure number
    for line in pop:
        n += 1
        if '|v.h|' in line:
            i += 1

        if len(line.split()) == 5 and 'POTENTIAL ENERGY VARIATION' not in line:
            if int(line.split()[1]) == i - 1:
                crt_pop = pop[n - 1:n - 1 + nstate]
                crt_pop[0] = crt_pop[0].split()[3]
                crt_pop = [float(i) for i in crt_pop]
                trj_pop.append(crt_pop)

            if i == maxstep and maxstep != 0:  # cutoff trajectories
                break

    ## Find kinetic and potential energy for each state and time step
    trj_kin = []  # kinetic energy list
    trj_pot = []  # potential energy list
    trj_tim = []  # time step list
    n = 0
    for line in eng:
        n += 1
        line = line.split()
        crt_time = float(line[0])
        crt_kin = float(line[-1]) - float(line[-2])
        crt_pot = line[1:nstate + 1]
        crt_pot = [float(i) for i in crt_pot]
        trj_kin.append(crt_kin)
        trj_pot.append(crt_pot)
        trj_tim.append(crt_time)

        if n - 1 == maxstep and maxstep != 0:  # cutoff trajectories
            break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab = len(trj_lab)
    len_pop = len(trj_pop)
    len_kin = len(trj_kin)
    len_pot = len(trj_pot)
    len_crd = len(trj_coord)
    nstep = np.amin([len_lab, len_pop, len_kin, len_pot, len_crd]).tolist()

    trj_lab = trj_lab[0:nstep]
    trj_pop = trj_pop[0:nstep]
    trj_kin = trj_kin[0:nstep]
    trj_pot = trj_pot[0:nstep]
    trj_coord = trj_coord[0:nstep]

    dtime = trj_tim[1]

    if pstep > 0:
        trj_lab = trj_lab[0:pstep]
        trj_pop = trj_pop[0:pstep]
        trj_kin = trj_kin[0:pstep]
        trj_pot = trj_pot[0:pstep]
        trj_coord = trj_coord[0:pstep]
        nstep = pstep
        crt_state = int(trj_lab[-1].split()[7])
        p_hop = trj_hop
        trj_hop = []
        for s in p_hop:
            if s <= pstep:
                trj_hop.append(s)

    return ntraj, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state


def read_raw_data_sharc(files):
    ## This function read sharc data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and
    ## append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj, f, maxstep, pindex, pthrhd = files

    with open('%s/output.dat' % f, 'r') as datfile:
        dat = datfile.read().splitlines()

    ## Find label and coordinates for each structure
    atom = []  # a list of atom
    natom = 0  # number of atoms
    nstate = 0  # number of states
    trj_lab = []  # label list
    trj_hop = []  # hopping event list
    trj_coord = []  # coordinates list
    trj_pop = []  # population list
    trj_kin = []  # kinetic energy list
    trj_pot = []  # potential energy list
    trj_state = []  # state list
    crt_label = ''  # current label
    crt_state = 0  # current state
    ini_state = 0  # initial state
    pre_state = 0  # previous state
    zero_energy = 0  # sharc ref energy
    dtime = 0  # time step
    i = 0  # count structure number
    pstep = 0  # count step to prune trajectory

    for n, line in enumerate(dat):
        if 'natom' in line:
            natom = int(line.split()[-1])
        if 'nstates_m ' in line:
            nstate = np.sum([(n + 1) * int(x) for n, x in enumerate(line.split()[1:])])
            nstate = int(nstate)
        if 'dtstep' in line:
            dtime = float(line.split()[-1]) / 20.6706868947804 * 0.5
        if 'ezero' in line:
            zero_energy = float(line.split()[-1])
        if '! Elements' in line:
            atom = dat[n + 1: n + 1 + natom]

        if '! 1 Hamiltonian' in line:
            i += 1
            crt_pot = dat[n + 1: n + 1 + nstate]
            crt_pot = np.array([p.split() for p in crt_pot]).astype(float)[:, ::2]
            crt_pot += zero_energy
            crt_pot = np.diag(crt_pot).tolist()
            trj_pot.append(crt_pot)

        if '! 5 Coefficients' in line:
            crt_pop = dat[n + 1: n + 1 + nstate]
            crt_pop = np.array([p.split() for p in crt_pop]).astype(float)
            crt_pop = np.sum(crt_pop ** 2, axis=1).tolist()
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
            crt_label = 'traj %d coord %d init %s' % (ntraj + 1, i, ini_state)
            if pre_state != crt_state:
                crt_hop = i
                hop_state = crt_state
                crt_label += ' state %d to %d CI' % (pre_state, hop_state)
                trj_hop.append(crt_hop)
                pre_state = crt_state
            else:
                crt_label += ' state %d' % crt_state
            trj_lab.append(crt_label)

        if '! 11 Geometry in a.u.' in line:
            xyz = dat[n + 1: n + 1 + natom]
            xyz = ['%-5s %14.8f %14.8f %14.8f' % (
                atom[n],
                float(x.split()[0]) * 0.529177,
                float(x.split()[1]) * 0.529177,
                float(x.split()[2]) * 0.529177
            ) for n, x in enumerate(xyz)]  # pick atom,x,y,z

            if pindex and pstep == 0:
                for k, p in enumerate(pindex):
                    p1, p2 = p
                    p3 = pthrhd[k]
                    stop, d = check_param(xyz, p1, p2, p3)
                    if stop:
                        pstep = i
                        break

            crt_coord = [crt_label] + xyz  # combine label with coordinates
            trj_coord.append(crt_coord)

        if i == maxstep and maxstep != 0:  # cutoff trajectories
            break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab = len(trj_lab)
    len_pop = len(trj_pop)
    len_kin = len(trj_kin)
    len_pot = len(trj_pot)
    len_crd = len(trj_coord)
    nstep = np.amin([len_lab, len_pop, len_kin, len_pot, len_crd]).tolist()

    trj_lab = trj_lab[0:nstep]
    trj_pop = trj_pop[0:nstep]
    trj_kin = trj_kin[0:nstep]
    trj_pot = trj_pot[0:nstep]
    trj_coord = trj_coord[0:nstep]

    if pstep > 0:
        trj_lab = trj_lab[0:pstep]
        trj_pop = trj_pop[0:pstep]
        trj_kin = trj_kin[0:pstep]
        trj_pot = trj_pot[0:pstep]
        trj_coord = trj_coord[0:pstep]
        nstep = pstep
        crt_state = int(trj_lab[-1].split()[7])
        p_hop = trj_hop
        trj_hop = []
        for s in p_hop:
            if s <= pstep:
                trj_hop.append(s)

    return ntraj, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state


def read_raw_data_fromage(files):
    ## This function read fromage data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and
    ## append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.
    ntraj, f, maxstep, pindex, pthrhd = files

    with open('%s/output.chk' % f, 'r') as datfile:
        dat = datfile.read().splitlines()

    with open('%s/geom_mol.xyz' % f, 'r') as xyzfile:
        xyz = xyzfile.read().splitlines()

    ## Find label and coordinates for each structure
    trj_lab = []  # label list
    trj_hop = []  # hopping event list
    trj_coord = []  # coordinates list
    trj_pop = []  # population list
    trj_kin = []  # kinetic energy list
    trj_pot = []  # potential energy list
    crt_state = 0  # current state
    ini_state = 0  # initial state
    pre_state = 0  # previous state

    ## Find energies and population
    checkline = dat[4].split()
    dtime = float(checkline[1])
    nstate = int(len(checkline[5:]) / 2)

    for n, line in enumerate(dat[3:]):
        line = line.split()
        crt_state = int(line[2]) - 1

        if n == 0:
            ini_state = crt_state
            pre_state = crt_state
        crt_label = 'traj %d coord %d init %s' % (ntraj + 1, n + 1, ini_state)

        if pre_state != crt_state:
            crt_hop = n + 1
            hop_state = crt_state
            crt_label += ' state %d to %d CI' % (pre_state, hop_state)
            trj_hop.append(crt_hop)
            pre_state = crt_state
        else:
            crt_label += ' state %d' % crt_state
        trj_lab.append(crt_label)

        crt_kin = float(line[4])
        trj_kin.append(crt_kin)

        crt_pot = line[6: nstate + 6]
        crt_pot = [float(x) for x in crt_pot]
        trj_pot.append(crt_pot)

        crt_pop = line[nstate + 6: nstate + nstate + 6]
        crt_pop = [float(x) for x in crt_pop]
        trj_pop.append(crt_pop)

        if n + 1 == maxstep and maxstep != 0:  # cutoff trajectories
            break

    ## Find coordinates

    natom = int(xyz[0])
    n = 0  # count line number
    m = 0  # count structure number
    pstep = 0  # count step to prune trajectory

    for _ in xyz:
        n += 1
        if n % (natom + 2) == 0:  # at the last line of each coordinates
            m += 1
            crt_coord = [trj_lab[m - 1]] + xyz[n - natom: n]  # combine label with coordinates
            trj_coord.append(crt_coord)

            if pindex and pstep == 0:
                for k, p in enumerate(pindex):
                    p1, p2 = p
                    p3 = pthrhd[k]
                    stop, d = check_param(xyz[n - natom: n], p1, p2, p3)
                    if stop:
                        pstep = m
                        break
            if m == maxstep and maxstep != 0:  # cutoff trajectories
                break

    ## Prune population, kinetic energy, potential energy, and coordinates list
    len_lab = len(trj_lab)
    len_pop = len(trj_pop)
    len_kin = len(trj_kin)
    len_pot = len(trj_pot)
    len_crd = len(trj_coord)
    nstep = np.amin([len_lab, len_pop, len_kin, len_pot, len_crd]).tolist()

    trj_lab = trj_lab[0:nstep]
    trj_pop = trj_pop[0:nstep]
    trj_kin = trj_kin[0:nstep]
    trj_pot = trj_pot[0:nstep]
    trj_coord = trj_coord[0:nstep]

    if pstep > 0:
        trj_lab = trj_lab[0:pstep]
        trj_pop = trj_pop[0:pstep]
        trj_kin = trj_kin[0:pstep]
        trj_pot = trj_pot[0:pstep]
        trj_coord = trj_coord[0:pstep]
        nstep = pstep
        crt_state = int(trj_lab[-1].split()[7])
        p_hop = trj_hop
        trj_hop = []
        for s in p_hop:
            if s <= pstep:
                trj_hop.append(s)

    return ntraj, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state


def read_pyrai2md(files):
    ## This function read data from provided folders
    ## This function is for parallelization
    ## subloops find crt_kin,crt_pot,crt_pop,crt_coord,crt_hop and
    ## append to trj_kin,trj_pot,trj_pop,trj_coord,trj_hop for each trajectory.
    ## natom, nstate, and dtime are supposed to be the same over all trajectories, so just pick up the last values.

    ntraj, f, pindex, pthrhd = files
    t = f.split('/')[-1]
    nstate = 0

    with open('%s/%s.sh.energies' % (f, t), 'r') as engfile:
        eng_h = engfile.read().splitlines()

    with open('%s/%s.sh.xyz' % (f, t), 'r') as xyzfile:
        xyz_h = xyzfile.read().splitlines()

    with open('%s/%s.md.energies' % (f, t), 'r') as engfile:
        eng_m = engfile.read().splitlines()

    with open('%s/%s.md.xyz' % (f, t), 'r') as xyzfile:
        xyz_m = xyzfile.read().splitlines()

    natom = int(xyz_m[0])
    n = 0
    pstep = 0
    stop = False
    for _ in xyz_m:
        n += 1
        if n % (natom + 2) == 0:  # at the last line of each coordinates
            pstep += 1
            if pindex:  # prune trajectory
                coord = xyz_m[n - natom:n]
                for m, p in enumerate(pindex):
                    p1, p2 = p
                    p3 = pthrhd[m]
                    stop, d = check_param(coord, p1, p2, p3)
                    if stop:
                        break
            if stop:
                break

    if len(eng_m) < 2:
        trj_init = []
        trj_init_t = []
        trj_init_p = []
        trj_final = []
        trj_final_t = []
        trj_final_p = []
        hstep = pstep
    else:
        nstate = len(eng_m[1].split()) - 4
        trj_init_p = eng_m[1].split()[4:nstate + 4]
        trj_init_p = [[float(i) for i in trj_init_p]]
        trj_final_p = eng_m[pstep - 1].split()[4:nstate + 4]  # prune trajectory
        trj_final_p = [[float(i) for i in trj_final_p]]

        trj_init = [xyz_m[1: 2 + natom]]
        lb = trj_init[0][0].split()
        trj_init[0][0] = 'traj %s coord 1 state %s' % (ntraj + 1, int(lb[4]) - 1)
        trj_init_t = [0]

        trj_final = [xyz_m[int((pstep - 1) * (natom + 2)) + 1: int(pstep * (natom + 2))]]
        lb = trj_final[0][0].split()
        trj_final[0][0] = 'traj %s coord %s state %s' % (ntraj + 1, int(lb[2]), int(lb[4]) - 1)
        trj_final_t = [int(lb[2])]
        hstep = int(trj_final_t[0])

    if len(eng_h) < 2:
        trj_hop = []
        trj_hop_t = []
        trj_hop_p = []
    else:
        ## Find population and label for each structure
        trj_hop_p = []  # pot energy
        for n, line in enumerate(eng_h):
            if 'time' in line:
                continue  # skip the title line

            if n == hstep - 1:  # prune trajectory
                break

            line = line.split()
            crt_pot = line[4:nstate + 4]
            crt_hop_p = [float(i) for i in crt_pot]
            trj_hop_p.append(crt_hop_p)

        ## Find coordinates
        nstate = len(eng_h[1].split()) - 4
        trj_hop = []  # coordinates list
        trj_hop_t = []
        natom = int(xyz_h[0])
        n = 0  # count line number
        m = 0  # count structure number
        for _ in xyz_h:
            n += 1
            if n % (natom + 2) == 0:  # at the last line of each coordinates
                m += 1
                lb = xyz_h[n - natom - 1].split()

                if int(lb[2]) > hstep:  # prune trajectory
                    break

                crt_label = 'traj %s coord %s state %s to %s CI' % (ntraj + 1, lb[2], int(lb[4]) - 1, int(lb[6]) - 1)
                crt_hop = [crt_label] + xyz_h[n - natom:n]  # combine label with coordinates
                trj_hop.append(crt_hop)
                trj_hop_t.append(lb[2])

    return ntraj, natom, nstate, trj_init, trj_final, trj_hop, trj_init_t, trj_final_t, trj_hop_t, trj_init_p, \
        trj_final_p, trj_hop_p

def RUNdiag(key_dict):
    ## This function run diagnosis for calculation results
    ## This function call count_data_lines
    ## This function print lines number of all required data for later analysis
    ## This function will save the index of the normally completed trajectories
    diag_func = {
        'molcas': count_data_lines_molcas,
        'newtonx': count_data_lines_newtonx,
        'sharc': count_data_lines_sharc,
        'fromage': count_data_lines_fromage,
    }

    cpus = key_dict['cpus']
    read_files = key_dict['read_files']
    prog = key_dict['prog']
    minstep = key_dict['minstep']

    procs = cpus
    input_val = []
    for n, f in enumerate(read_files):
        input_val.append([n, f])

    result = [[] for _ in range(len(input_val))]

    if (len(input_val)) < procs:
        procs = len(input_val)
    sys.stdout.write('CPU: %3d Checking data: \r' % procs)
    pool = multiprocessing.Pool(processes=procs)
    for ntraj, val in enumerate(pool.imap_unordered(diag_func[prog], input_val)):
        ntraj += 1
        p, name, nlog, nenergy, nxyz = val
        result[p] = [name, nlog, nenergy, nxyz]
        sys.stdout.write(
            'CPU: %3d Checking data: %6.2f%% %d/%d\r' % (procs, ntraj * 100 / (len(input_val)), ntraj, len(input_val)))

    select = []

    print('\nDiagnosis results\n%-20s%12s%12s%12s\n' % ('Name', '.log', '.md.energy', '.md.xyz'))
    for i in result:
        print('%-20s%12d%12d%12d' % (i[0], i[1], i[2], i[3]))
        if i[1] == i[2] == i[3] >= minstep:
            select.append(i[0])

    if len(select) > 0:
        print('\nSaving the index of the normally completed trajectory to file: complete\n')
        complete = ''
        for f in select:
            complete += '%s\n' % f
        with open('complete', 'w') as out:
            out.write(complete)
    else:
        print('\nNone of the trajectories complete normally')


def RUNcheck(key_dict):
    ## This function check energy conservation for calculation results
    ## This function call read_raw_data
    ## This function will save the index of the energy-conserved trajectories
    read_func = {
        'molcas': read_raw_data_molcas,
        'newtonx': read_raw_data_newtonx,
        'sharc': read_raw_data_sharc,
        'fromage': read_raw_data_fromage,
    }

    cpus = key_dict['cpus']
    read_files = key_dict['read_files']
    prog = key_dict['prog']
    maxstep = key_dict['maxstep']
    maxdrift = key_dict['maxdrift']
    pindex = key_dict['pindex']
    pthrhd = key_dict['pthrhd']

    ## A parallelized loop goes over all trajectories to find
    ## trj_kin,trj_pot,trj_pop,trj_coord,trj_hop in each trajectory and appends to
    ## kin, pot, pop, coord, hop in the main dictionary.
    input_val = []
    for n, f in enumerate(read_files):
        input_val.append([n, f, maxstep, pindex, pthrhd])

    if (len(input_val)) < cpus:
        cpus = len(input_val)

    drift = [[] for _ in range(len(input_val))]
    sys.stdout.write('CPU: %3d Reading data: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for ntraj, val in enumerate(pool.imap_unordered(read_func[prog], input_val)):
        ntraj += 1
        p, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state = val

        ## compute total energy
        crt_label = [int(x.split()[7]) for x in trj_lab]  # state number is the 8th data
        crt_index = [x for x in range(len(crt_label))]
        total_energy = np.array(trj_kin) + np.array(trj_pot)[crt_index, crt_label]
        drift[p] = total_energy - total_energy[0]

        sys.stdout.write(
            'CPU: %3d Reading data: %6.2f%% %d/%d\r' % (cpus, ntraj * 100 / (len(input_val)), ntraj, len(input_val)))

    select = []

    print('\nEnergy conservation results\n%-20s%12s%12s%12s\n' % ('Name', '  MAE  ', ' MAXABS ', '  Mean  '))
    for n, i in enumerate(drift):
        name = read_files[n]
        mae_drift = np.mean(np.abs(i))
        maxabs_drift = np.max(np.abs(i))
        mean_drift = np.mean(i)

        print('%-20s%12.4f%12.4f%12.4f' % (name, float(mae_drift), float(maxabs_drift), float(mean_drift)))
        if maxabs_drift < maxdrift:
            select.append(name)

    if len(drift) > 0:
        print('\nSaving the energy drift to file: energy_drift.txt\n')
        drift_data = ''
        for f in drift:
            drift_data += '%s\n' % ' '.join(['%16.8f' % x for x in f])
        with open('energy_drift.txt', 'w') as out:
            out.write(drift_data)

    if len(select) > 0:
        print('\nSaving the index of the energy-conserved trajectory to file: conserved\n')
        conserved = ''
        for f in select:
            conserved += '%s\n' % f
        with open('conserved', 'w') as out:
            out.write(conserved)
    else:
        print('\nNone of the trajectories complete normally')


def RUNread(key_dict):
    ## This function read data from calculation folders
    ## This function call read_raw_data
    ## This function return a dictionary for
    ## lists of natom, nstate, nstep, dtime, kin, pot, pop, label, coord, hop, prod

    read_func = {
        'molcas': read_raw_data_molcas,
        'newtonx': read_raw_data_newtonx,
        'sharc': read_raw_data_sharc,
        'fromage': read_raw_data_fromage,
    }

    ## initialize variables
    title = key_dict['title']
    cpus = key_dict['cpus']
    read_files = key_dict['read_files']
    save_traj = key_dict['save_traj']
    prog = key_dict['prog']
    maxstep = key_dict['maxstep']
    pindex = key_dict['pindex']
    pthrhd = key_dict['pthrhd']
    natom = 0
    nstate = 0
    ntraj = 0
    dtime = 0

    ## A parallelized loop goes over all trajectories to find
    ## trj_kin,trj_pot,trj_pop,trj_coord,trj_hop in each trajectory and appends to
    ## kin, pot, pop, coord, hop in the main dictionary.
    input_val = []
    for n, f in enumerate(read_files):
        input_val.append([n, f, maxstep, pindex, pthrhd])
    kin = [[] for _ in range(len(input_val))]
    pot = [[] for _ in range(len(input_val))]
    pop = [[] for _ in range(len(input_val))]
    label = [[] for _ in range(len(input_val))]
    coord = [[] for _ in range(len(input_val))]
    hop = [[] for _ in range(len(input_val))]
    last = []
    if (len(input_val)) < cpus:
        cpus = len(input_val)
    sys.stdout.write('CPU: %3d Reading data: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for ntraj, val in enumerate(pool.imap_unordered(read_func[prog], input_val)):
        ntraj += 1
        p, natom, nstate, nstep, dtime, trj_kin, trj_pot, trj_pop, trj_lab, trj_coord, trj_hop, crt_state = val

        ## create a list to classify trajectories according to their final state
        if ntraj == 1:
            last = [[] for _ in range(nstate)]

        ## send data
        last[crt_state].append(p + 1)
        kin[p] = trj_kin
        pot[p] = trj_pot
        pop[p] = trj_pop
        label[p] = trj_lab
        coord[p] = trj_coord
        hop[p] = trj_hop
        sys.stdout.write(
            'CPU: %3d Reading data: %6.2f%% %d/%d\r' % (cpus, ntraj * 100 / (len(input_val)), ntraj, len(input_val)))

    ## sort trajectory indices
    hop = [sorted(i) for i in hop]
    last = [sorted(i) for i in last]

    length = []
    for p in pop:
        length.append(len(p))
    nstep = int(np.amax(length))

    main_dict = {
        'title': title,
        'natom': natom,
        'nstate': nstate,
        'ntraj': ntraj,
        'nstep': nstep,
        'dtime': dtime,
        'kin': kin,
        'pot': pot,
        'pop': pop,
        'label': label,
        'coord': coord,
        'hop': hop,
        'last': last,
    }

    if save_traj == 1:
        print('\nSave the trajectory to %s.json\n' % title)
        with open('%s.json' % title, 'w') as indata:
            json.dump(main_dict, indata)
    else:
        print('\nSkip trajectory saving\n')

    # print('title,natom,nstate,ntraj,dtime')
    # print(title,natom,nstate,ntraj,dtime)
    # print('len(kin),len(pot),len(pop),len(coord)')
    # print(len(kin),len(pot),len(pop),len(coord))
    # print(kin,pot)
    # print('1',len(pot[0]),len(pop[0]),'1')
    # print(len(kin[0]),len(pot[0][0]),len(pop[0][0]),len(coord[0]))
    # exit()

    geom_i = {}
    time_i = {}
    pot_i = {}
    geom_f = {}
    time_f = {}
    pot_f = {}
    geom_h = {}
    time_h = {}
    pot_h = {}
    state_info = ''
    ## output CI and products coordinates according to final state
    print('\nNumber of Trajectories: %5d\nNumber of States:       %5d\n' % (ntraj, nstate))

    for i_state, traj_index in enumerate(last):  # trajectory index in each state
        init_snapshot = ''
        last_snapshot = ''
        hop_snapshot = ''
        init_geom = {}
        last_geom = {}
        hop_geom = {}
        init_pot = {}
        last_pot = {}
        hop_pot = {}
        init_time = {}
        last_time = {}
        hop_time = {}
        for i_traj in traj_index:
            init_snapshot += format1(natom, coord[i_traj - 1][0])
            init_geom[i_traj] = format3(natom, coord[i_traj - 1][0])
            init_pot[i_traj] = pot[i_traj - 1][0]
            init_time[i_traj] = '0'
            last_snapshot += format1(natom, coord[i_traj - 1][-1])
            last_geom[i_traj] = format3(natom, coord[i_traj - 1][-1])
            last_pot[i_traj] = pot[i_traj - 1][-1]
            last_time[i_traj] = label[i_traj - 1][-1].split()[3]  # step is the 4th data
            # len(pot[i_traj - 1])

            hop_index = hop[i_traj - 1]
            if len(hop_index) == 0:
                continue

            for i_hop in hop_index:
                if len(coord[i_traj - 1]) >= i_hop:
                    hop_snapshot += format1(natom, coord[i_traj - 1][i_hop - 1])

            hop_geom[i_traj] = format3(natom, coord[i_traj - 1][hop_index[-1] - 1])
            hop_pot[i_traj] = pot[i_traj - 1][hop_index[-1] - 1]
            hop_time[i_traj] = label[i_traj - 1][hop_index[-1] - 1].split()[3]  # step is the 4th data

        with open('Ini.S%d.xyz' % i_state, 'w') as outxyz:
            outxyz.write(init_snapshot)

        with open('Fin.S%d.xyz' % i_state, 'w') as outxyz:
            outxyz.write(last_snapshot)

        with open('Hop.S%d.xyz' % i_state, 'w') as outxyz:
            outxyz.write(hop_snapshot)

        geom_i[i_state] = init_geom
        time_i[i_state] = init_time
        pot_i[i_state] = init_pot
        geom_f[i_state] = last_geom
        time_f[i_state] = last_time
        pot_f[i_state] = last_pot
        geom_h[i_state] = hop_geom
        time_h[i_state] = hop_time
        pot_h[i_state] = hop_pot

        index_range = redindex(last[i_state])
        print('\nState %5d:  %5d  Write: Fin.S%d.xyz and Hop.S%d.xyz Range: %s' % (
            i_state, len(last[i_state]), i_state, i_state, index_range))

        state_info += '\nState %5d:  %5d Write: Fin.S%d.xyz and Hop.S%d.xyz\n\n%s\n' % (
            i_state, len(last[i_state]), i_state, i_state, format2(index_range))

    geom_dict = {
        'natom': natom,
        'nstate': nstate,
        'ntraj': ntraj,
        'dtime': dtime,
        'hop': geom_h,
        'final': geom_f,
        'init': geom_i,
        'init_t': time_i,
        'final_t': time_f,
        'hop_t': time_h,
        'init_p': pot_i,
        'final_p': pot_f,
        'hop_p': pot_h,
    }

    print('\nSave the hopping and final snapshots to Geom-%s.json\n' % title)
    with open('Geom-%s.json' % title, 'w') as indata:
        json.dump(geom_dict, indata)

        log_info = """
=> Trajectory summary
-------------------------------------
Number of atoms:            %-10s
Number of states:           %-10s
Number of trajectories:     %-10s
Time step (a.u.):           %-10s
%s
        """ % (natom, nstate, ntraj, dtime, state_info)

        with open('%s.traj.log' % title, 'a') as trajlog:
            trajlog.write(log_info)

    return main_dict, geom_dict


def RUNpop(key_dict):
    ## This function compute state population and hop energy gap
    ## This function unpack geom dictionary from json
    ## This function save the computed values to text
    title = key_dict['title']
    select = key_dict['select']
    save_data = key_dict['save_data']

    if os.path.exists('%s.json' % title):
        print('\nLoad trajectories from %s.json' % title)
        with open('%s.json' % title, 'r') as indata:
            raw_data = json.load(indata)
        natom = raw_data['natom']
        nstate = raw_data['nstate']
        ntraj = raw_data['ntraj']
        dtime = raw_data['dtime']
        last = raw_data['last']
        state_info = ''
        for i_state, traj_state_index in enumerate(last):
            index_range = redindex(traj_state_index)
            print('\nState %5d:  %5d Range: %s' % (i_state, len(traj_state_index), index_range))
            state_info += '\nState %5d:  %5d Range:\n\n%s\n' % (i_state, len(traj_state_index), format2(index_range))

        log_info = """
=> Trajectory summary
-------------------------------------
Number of atoms:            %-10s
Number of states:           %-10s
Number of trajectories:     %-10s
Time step (a.u.):           %-10s
%s
            """ % (natom, nstate, ntraj, dtime, state_info)

        with open('%s.traj.log' % title, 'a') as trajlog:
            trajlog.write(log_info)

    else:
        print('\nRead data from calculation folders')
        raw_data, geom_dict = RUNread(key_dict)
        last = raw_data['last']
        nstate = raw_data['nstate']
        ntraj = raw_data['ntraj']
        dtime = raw_data['dtime']

    nstep = raw_data['nstep']
    kin: list = raw_data['kin']
    pot: list = raw_data['pot']
    pop: list = raw_data['pop']
    label: list = raw_data['label']
    hop: list = raw_data['hop']

    # compute pop and hop energy gap
    log_info = ''
    repair = []
    skipex = []
    skipnan = []
    skipexcd = []
    kept = []
    naverage = 0
    all_lab = []
    all_kin = []
    all_pot = []
    all_tot = []
    all_pop = []
    avg_kin = np.zeros(nstep)
    avg_pot = np.zeros([nstep, nstate])
    avg_tot = np.zeros([nstep, nstate])
    avg_pop = np.zeros([nstep, nstate])
    hop_energy = ''

    if len(select) == 0:
        traj_index = [x + 1 for x in range(ntraj)]
        print('\nUse all trajectories for population analysis: %s\n' % ntraj)
    else:
        traj_index = select
        print('\nUse selected population analysis: %s\n' % (len(traj_index)))

    for i_traj in traj_index:
        exceed = 0
        crt_lab = [x.split()[7] for x in label[i_traj - 1]]  # state number is the 8th data
        crt_kin = kin[i_traj - 1]
        crt_pot = pot[i_traj - 1]
        crt_pop = pop[i_traj - 1]
        # expand kinetic energy to each state
        crt_tot = (np.array([[x for _ in range(nstate)] for x in kin[i_traj - 1]]) + np.array(crt_pot)).tolist()

        ## check if population is nan
        if True in np.isnan(crt_pop):
            skipnan.append(i_traj)
            continue

        ## check if population exceed 0-1
        if np.amax(crt_pop) > 1.01 or np.amin(crt_pop) < -0.01:
            exceed = 1

        if exceed == 1:
            skipexcd.append(i_traj)
            continue

        ## compute the missing step
        fstate = int(crt_lab[-1])
        dstep = nstep - len(kin[i_traj - 1])
        if dstep != 0:  # complete the missing part by repeating the last step for trajectories loaded in ground state
            if fstate == 0:
                repair.append(i_traj)
                crt_lab = crt_lab + [crt_lab[-1] for _ in range(dstep)]
                crt_kin = crt_kin + [crt_kin[-1] for _ in range(dstep)]
                crt_pot = crt_pot + [crt_pot[-1] for _ in range(dstep)]
                crt_tot = crt_tot + [crt_tot[-1] for _ in range(dstep)]
                crt_pop = crt_pop + [crt_pop[-1] for _ in range(dstep)]
            else:
                skipex.append(i_traj)
                continue

        kept.append(i_traj)
        avg_kin += np.array(crt_kin)
        avg_pot += np.array(crt_pot)
        avg_tot += np.array(crt_tot)
        avg_pop += np.array(crt_pop)

        all_lab.append(crt_lab)
        all_kin.append(crt_kin)
        all_pot.append(crt_pot)
        all_tot.append(crt_tot)
        all_pop.append(crt_pop)

        naverage += 1

        # compute hop energy gap
        crt_hop = hop[i_traj - 1]
        if len(crt_hop) > 0:
            for i_hop in crt_hop:
                init_state = int(label[i_traj - 1][i_hop - 1].split()[7])
                final_state = int(label[i_traj - 1][i_hop - 1].split()[9])
                gap = np.abs(crt_pot[i_hop - 1][init_state] - crt_pot[i_hop - 1][final_state]) * 27.211
                hop_energy += '%-5d %5d %5d %12.4f\n' % (i_traj, init_state, final_state, gap)

    avg_kin /= naverage
    avg_pot /= naverage
    avg_tot /= naverage
    avg_pop /= naverage

    cap_kin = '%24s' % 'Ekin'
    cap_pot = ''.join(['%24s' % ('Epot %s' % o) for o in range(nstate)])
    cap_tot = ''.join(['%24s' % ('Etot %s' % o) for o in range(nstate)])
    cap_pop = ''.join(['%24s' % ('Pop %s' % o) for o in range(nstate)])
    average = '%-20d%s%s%s%s\n' % (dtime, cap_kin, cap_pot, cap_tot, cap_pop)
    for a in range(nstep):
        cap_pot = ''.join(['%24.16f' % x for x in avg_tot[a]])
        cap_tot = ''.join(['%24.16f' % x for x in avg_tot[a]])
        cap_pop = ''.join(['%24.16f' % x for x in avg_pop[a]])
        average += '%-20d%24.16f%s%s%s\n' % (a, avg_kin[a], cap_pot, cap_tot, cap_pop)

    average_info = '\n'
    average_info += 'Repair trajectories:   %5d of %5d\n\n%s\n' % (
        len(repair), len(traj_index), format2(redindex(repair)))
    average_info += 'Skip trajectories early stopped in excited states:   %5d of %5d\n\n%s\n' % (
        len(skipex), len(traj_index), format2(redindex(skipex)))
    average_info += 'Skip trajectories with nan population:   %5d of %5d\n\n%s\n' % (
        len(skipnan), len(traj_index), format2(redindex(skipnan)))
    average_info += 'Skip trajectories with population exceeding 0-1:   %5d of %5d\n\n%s\n' % (
        len(skipexcd), len(traj_index), format2(redindex(skipexcd)))
    average_info += 'Averaged trajectories: %5d (%5d - %5d - %5d - %5d)\n\n%s\n' % (
        naverage, len(traj_index), len(skipex), len(skipnan), len(skipexcd), format2(redindex(kept)))
    average_info += 'Save population data: average-%s.dat\n' % title
    average_info += 'Save hop energy data: hop-energy-%s.dat\n' % title
    average_info += '\nUpdated state index information\n'

    print(average_info)

    skip_index = skipex + skipnan + skipexcd
    for i_state, traj_state_index in enumerate(last):
        new_index = []
        for idx in traj_state_index:
            if idx not in skip_index and idx in traj_index:
                new_index.append(idx)
        index_range = redindex(new_index)
        print('\nState %5d:  %5d Range: %s' % (i_state, len(new_index), index_range))
        average_info += '\nState %5d:  %5d Range: \n\n%s\n' % (i_state, len(new_index), format2(index_range))

    log_info += average_info

    with open('average-%s.dat' % title, 'w') as outsum:
        outsum.write(average)

    with open('hop-energy-%s.dat' % title, 'w') as outsum:
        outsum.write(hop_energy)

        if save_data >= 1:
            average_info = 'Save state populations:  state-pop-%s.json\n' % title

            print(average_info)
            log_info += average_info

            with open('state-pop-%s.json' % title, 'w') as outpop:
                json.dump(all_pop, outpop)

        if save_data >= 2:
            average_info = 'Save energy profiles:    energy-profile-%s.json\n' % title

            print(average_info)
            log_info += average_info

            with open('energy-profile-%s.json' % title, 'w') as outtot:
                json.dump([all_lab, all_kin, all_pot, all_tot], outtot)

    with open('%s.traj.log' % title, 'a') as trajlog:
        trajlog.write(log_info)

def classify_prep(order, geom, step, pot, snapshot_type, classify_state, param_list, thrhd, ref_coord, select):
    input_val = []
    ntraj_list = []
    time_list = []
    gap_list = []
    if len(geom[classify_state]) == 0:
        exit('\nNo %s snapshot at state %s, please change classify or classify_state' % (snapshot_type, classify_state))

    i_coord = -1
    for ntraj in geom[classify_state].keys():  # geom_f[classify_state] is a dict
        if len(select) > 0:
            if int(ntraj) not in select:
                continue
        i_coord += 1
        coord = format4(geom[classify_state][ntraj])
        ntraj_list.append(ntraj)
        time_list.append(step[classify_state][ntraj])
        gap_list.append(' '.join(['%24.16f' % x for x in pot[classify_state][ntraj]]))
        for i_param, param in enumerate(param_list):
            input_val.append([order, i_coord, i_param, coord, param, thrhd[i_param], ref_coord])

    param = [[[] for _ in param_list] for _ in range(i_coord + 1)]

    return param, input_val, ntraj_list, time_list, gap_list

def classify_output(title, snapshot_type, param, ntraj_list, ext, time, gap):
    output_f = ''
    structure_f = {}
    param = np.array(param)
    for n, p in enumerate(param):
        ntraj = ntraj_list[n]
        value = ''.join(['%12.4f' % x[0] for x in p])
        label = ''.join(['%s' % int(x[1]) for x in p])
        if len(time) > 0:
            value += ' %12s ' % time[n]

        if len(gap) > 0:
            value += ' %s ' % gap[n]

        output_f += '%-5s %s\n' % (ntraj, value)

        if label in structure_f.keys():
            structure_f[label].append(ntraj)
        else:
            structure_f[label] = [ntraj]

    print('\nSave parameters for %s snapshot to param-%s.%s' % (snapshot_type, title, ext))
    with open('param-%s.%s' % (title, ext), 'w') as out:
        out.write(output_f)

    print('\nSummary: %s snapshot' % snapshot_type)
    print('Label  Ntraj    %')
    for label, ntraj in structure_f.items():
        print('%s %5d      %5.2f' % (label, len(ntraj), len(ntraj) / len(param)))

def RUNclassify(key_dict):
    ## This function classify hopping and final structures
    ## This function unpack geom dictionary from json
    ## This function save the computed values to text
    title = key_dict['title']
    cpus = key_dict['cpus']
    classify = key_dict['classify']
    classify_state = key_dict['classify_state']
    output_atom = key_dict['output_atom']
    align = key_dict['align']
    align_core = key_dict['align_core']
    select = key_dict['select']
    ref_geom = key_dict['ref_geom']
    param = key_dict['param']
    thrhd = key_dict['thrhd']
    param_list = Paramread(param)

    if ref_geom is not None:
        ref_coord = Refread(ref_geom)[0]
    else:
        ref_coord = []

    if len(thrhd) != len(param_list):
        thrhd = [0 for _ in range(len(param_list))]

    if os.path.exists('Geom-%s.json' % title):
        print('\nRead snapshots from Geom-%s.json' % title)
        with open('Geom-%s.json' % title, 'r') as indata:
            geom_dict = json.load(indata)
        natom = geom_dict['natom']
        nstate = geom_dict['nstate']
        ntraj = geom_dict['ntraj']
        dtime = geom_dict['dtime']

        geom_i = {}
        for key, val in geom_dict['init'].items():
            geom_i[int(key)] = val

        geom_f = {}
        for key, val in geom_dict['final'].items():
            geom_f[int(key)] = val

        geom_h = {}
        for key, val in geom_dict['hop'].items():
            geom_h[int(key)] = val

        step_i = {}
        for key, val in geom_dict['init_t'].items():
            step_i[int(key)] = val

        step_f = {}
        for key, val in geom_dict['final_t'].items():
            step_f[int(key)] = val

        step_h = {}
        for key, val in geom_dict['hop_t'].items():
            step_h[int(key)] = val

        pot_i = {}
        for key, val in geom_dict['init_p'].items():
            pot_i[int(key)] = val

        pot_f = {}
        for key, val in geom_dict['final_p'].items():
            pot_f[int(key)] = val

        pot_h = {}
        for key, val in geom_dict['hop_p'].items():
            pot_h[int(key)] = val

        state_info = ''
        for n, g in enumerate(geom_f):
            print('\nState %5d:  %5d' % (n, len(geom_f[g])))
            state_info += '\nState %5d:  %5d' % (n, len(geom_f[g]))

        log_info = """
=> Trajectory summary
-------------------------------------
Number of atoms:            %-10s
Number of states:           %-10s
Number of trajectories:     %-10s
Time step (a.u.):           %-10s
%s
        """ % (natom, nstate, ntraj, dtime, state_info)

        with open('%s.traj.log' % title, 'a') as trajlog:
            trajlog.write(log_info)

    else:
        print('\nRead data from calculation folders')
        raw_data, geom_dict = RUNread(key_dict)
        geom_i = geom_dict['init']
        geom_f = geom_dict['final']
        geom_h = geom_dict['hop']
        step_i = geom_dict['init_t']
        step_f = geom_dict['final_t']
        step_h = geom_dict['hop_t']
        pot_i = geom_dict['init_p']
        pot_f = geom_dict['final_p']
        pot_h = geom_dict['hop_p']

    ## compute geometrical parameters
    input_val_i = []
    param_i = []
    ntraj_i = []
    time_i = []
    gap_i = []
    if classify == 'all' or classify == 'init':
        param_i, input_val_i, ntraj_i, time_i, gap_i = classify_prep(
            0, geom_i, step_i, pot_i, 'initial', classify_state, param_list, thrhd, ref_coord, select
        )
        if align:
            align_mol('Ini.S%d.xyz' % classify_state, align_core, output_atom, cpus)

    input_val_f = []
    param_f = []
    ntraj_f = []
    time_f = []
    gap_f = []
    if classify == 'all' or classify == 'final':
        param_f, input_val_f, ntraj_f, time_f, gap_f = classify_prep(
            1, geom_f, step_f, pot_f, 'final', classify_state, param_list, thrhd, ref_coord, select
        )
        if align:
            align_mol('Fin.S%d.xyz' % classify_state, align_core, output_atom, cpus)

    input_val_h = []
    param_h = []
    ntraj_h = []
    time_h = []
    gap_h = []
    if classify == 'all' or classify == 'hop':
        param_h, input_val_h, ntraj_h, time_h, gap_h = classify_prep(
            2, geom_h, step_h, pot_h, 'hop', classify_state, param_list, thrhd, ref_coord, select
        )
        if align:
            align_mol('Hop.S%d.xyz' % classify_state, align_core, output_atom, cpus)

    input_val = input_val_i + input_val_f + input_val_h
    param_all = [param_i, param_f, param_h]

    if (len(input_val)) < cpus:
        cpus = len(input_val)

    sys.stdout.write('CPU: %3d Computing parameters: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for p, val in enumerate(pool.imap_unordered(compute_para, input_val)):
        p += 1
        i_traj, i_geom, i_param, geom_param, geom_type = val
        param_all[i_traj][i_geom][i_param] = [geom_param, geom_type]
        sys.stdout.write(
            'CPU: %3d Computing parameters: %6.2f%% %d/%d\r' % (cpus, p * 100 / (len(input_val)), p, len(input_val)))

    ## save geometrical parameters
    if len(param_all[0]) > 0:
        classify_output(title, 'init', param_all[0], ntraj_i, 'S%s.ini' % classify_state, time_i, gap_i)

    if len(param_all[1]) > 0:
        classify_output(title, 'final', param_all[1], ntraj_f, 'S%s.fin' % classify_state, time_f, gap_f)

    if len(param_all[2]) > 0:
        classify_output(title, 'hop', param_all[2], ntraj_h, 'S%s.hop' % classify_state, time_h, gap_h)


def RUNcompute(key_dict):
    ## This function compute geometrical parameter for plot
    ## This function unpack main dictionary from raw_data
    ## This function call compute_para for parallelization
    title = key_dict['title']
    cpus = key_dict['cpus']
    select = key_dict['select']
    ref_geom = key_dict['ref_geom']
    param = key_dict['param']
    param_list = Paramread(param)

    if ref_geom is not None:
        ref_coord = Refread(ref_geom)[0]
    else:
        ref_coord = []

    if os.path.exists('%s.json' % title):
        print('\nLoad trajectories from %s.json' % title)
        with open('%s.json' % title, 'r') as indata:
            raw_data = json.load(indata)
        natom = raw_data['natom']
        nstate = raw_data['nstate']
        ntraj = raw_data['ntraj']
        dtime = raw_data['dtime']
        last = raw_data['last']
        state_info = ''
        for n, g in enumerate(last):
            print('\nState %5d:  %5d' % (n, len(g)))
            state_info += '\nState %5d:  %5d' % (n, len(g))

        log_info = """
    => Trajectory summary
    -------------------------------------
    Number of atoms:            %-10s
    Number of states:           %-10s
    Number of trajectories:     %-10s
    Time step (a.u.):           %-10s
    %s
            """ % (natom, nstate, ntraj, dtime, state_info)

        with open('%s.traj.log' % title, 'a') as trajlog:
            trajlog.write(log_info)

    else:
        print('\nRead data from calculation folders')
        raw_data, geom_dict = RUNread(key_dict)

    ntraj = raw_data['ntraj']
    kin: list = raw_data['kin']
    pot: list = raw_data['pot']
    label: list = raw_data['label']
    hop: list = raw_data['hop']
    coord: list = raw_data['coord']

    if len(select) == 0:
        traj_index = [x + 1 for x in range(ntraj)]
    else:
        traj_index = select

    ## compute geometrical parameters for selected trajectories
    traj_info = '\nSelect trajectory:\n\n %s\n' % (format2(redindex(traj_index)))
    print('\nSelect trajectory: %s\n' % (redindex(traj_index)))
    plot_para = []
    plot_etot = []
    plot_hop = []
    plot_label = []
    input_val = []

    print('\nPrepare plot data...\n')
    for i_traj, traj_idx in enumerate(traj_index):
        plot_hop.append(hop[traj_idx - 1])
        plot_label.append(label[traj_idx - 1])
        trj_para = []
        trj_etot = []
        for i_geom, geom in enumerate(coord[traj_idx - 1]):  # for each coordinates
            geom = geom[1:]  # remove the label
            crt_label = label[traj_idx - 1][i_geom]
            crt_state = int(crt_label.split()[7])  # state number is the 8th data
            crt_kin = kin[traj_idx - 1][i_geom]
            crt_pot = pot[traj_idx - 1][i_geom][crt_state]
            crt_tot = crt_kin + crt_pot
            trj_para.append([0 for _ in range(len(param_list))])
            trj_etot.append(crt_tot)
            for i_param, param in enumerate(param_list):  # for each parameter
                input_val.append([i_traj, i_geom, i_param, geom, param, 0, ref_coord])

        plot_para.append(trj_para)
        plot_etot.append(trj_etot)

    if (len(input_val)) < cpus:
        cpus = len(input_val)

    sys.stdout.write('CPU: %3d Saving plot data: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for p, val in enumerate(pool.imap_unordered(compute_para, input_val)):
        p += 1
        i_traj, i_geom, i_param, geom_param, geom_type = val
        plot_para[i_traj][i_geom][i_param] = geom_param
        sys.stdout.write(
            'CPU: %3d Saving plot data: %6.2f%% %d/%d\r' % (cpus, p * 100 / (len(input_val)), p, len(input_val)))

    plot_data = {
        'title': title,
        'data': plot_para,
        'energy': plot_etot,
        'hop': plot_hop,
        'label': plot_label,
    }

    with open('plot-%s.json' % title, 'w') as outxyz:
        print('\nPlot data saved in plot-%s.json' % title)
        json.dump(plot_data, outxyz)

    traj_info += '\nPlot data saved in plot-%s.json' % title


def RUNfetch(key_dict):
    ## This function fetch selected trajectories
    ## This function unpack main dictionary from raw_data
    ## This function save state, kinetic and potential energy, population
    ## in .dat and label coordinates in .xyz when follow != None
    title = key_dict['title']
    select = key_dict['select']

    if os.path.exists('%s.json' % title):
        print('\nLoad trajectories from %s.json' % title)
        with open('%s.json' % title, 'r') as indata:
            raw_data = json.load(indata)
        natom = raw_data['natom']
        nstate = raw_data['nstate']
        ntraj = raw_data['ntraj']
        dtime = raw_data['dtime']
        last = raw_data['last']
        state_info = ''
        for n, g in enumerate(last):
            print('\nState %5d:  %5d' % (n, len(g)))
            state_info += '\nState %5d:  %5d' % (n, len(g))

        log_info = """
    => Trajectory summary
    -------------------------------------
    Number of atoms:            %-10s
    Number of states:           %-10s
    Number of trajectories:     %-10s
    Time step (a.u.):           %-10s
    %s
            """ % (natom, nstate, ntraj, dtime, state_info)

        with open('%s.traj.log' % title, 'a') as trajlog:
            trajlog.write(log_info)

    else:
        print('\nRead data from calculation folders')
        raw_data, geom_dict = RUNread(key_dict)

    natom = raw_data['natom']
    nstate = raw_data['nstate']
    dtime = raw_data['dtime']
    kin: list = raw_data['kin']
    pot: list = raw_data['pot']
    pop: list = raw_data['pop']
    coord: list = raw_data['coord']
    label: list = raw_data['label']

    if len(select) == 0:
        exit('\nNo selected trajectories')

    print('\nSelect trajectory: %s' % (redindex(select)))
    trajs_info = '\nSelect trajectory:\n\n%s\n' % (format2(redindex(select)))
    for i_traj in select:
        print('Traj: %5d Write: select-%s-%s.xyz and select-%s-%s.dat' % (i_traj, title, i_traj, title, i_traj))
        trajs_info += 'Traj: %5d Write: select-%s-%s.xyz and select-%s-%s.dat' % (i_traj, title, i_traj, title, i_traj)
        select_traj = ''
        for geom in coord[i_traj - 1]:
            select_traj += format1(natom, geom)

        with open('select-%s-%s.xyz' % (title, i_traj), 'w') as outxyz:
            outxyz.write(select_traj)

        cap_pot = ''.join(['%24s' % ('Epot %s' % s) for s in range(nstate)])
        cap_pop = ''.join(['%24s' % ('Pop %s' % s) for s in range(nstate)])
        select_traj = '%-20d%5s%24s%s%s\n' % (dtime, 'state', 'Ekin', cap_pot, cap_pop)
        crt_label = label[i_traj - 1]
        crt_kin = np.array(kin[i_traj - 1])
        crt_pot = np.array(pot[i_traj - 1])
        crt_pop = np.array(pop[i_traj - 1])
        for i_geom, _ in enumerate(coord[i_traj - 1]):
            crt_state = int(crt_label[i_geom].split()[7])  # state number is the 8th data
            cap_kin = '%24.16f' % crt_kin[i_geom]
            cap_pot = ''.join(['%24.16f' % x for x in crt_pot[i_geom]])
            cap_pop = ''.join(['%24.16f' % x for x in crt_pop[i_geom]])
            select_traj += '%-20d%5d%s%s%s\n' % (i_geom, crt_state, cap_kin, cap_pot, cap_pop)

        with open('select-%s-%s.dat' % (title, i_traj), 'w') as outxyz:
            outxyz.write(select_traj)

def RUNreadpmd(key_dict):
    title = key_dict['title']
    cpus = key_dict['cpus']
    read_files = key_dict['read_files']
    pindex = key_dict['pindex']
    pthrhd = key_dict['pthrhd']

    ntraj = 0
    atoms = []
    states = []
    input_val = []
    for n, f in enumerate(read_files):
        input_val.append([n, f, pindex, pthrhd])

    init = [[] for _ in range(len(input_val))]
    final = [[] for _ in range(len(input_val))]
    hop = [[] for _ in range(len(input_val))]
    init_t = [[] for _ in range(len(input_val))]
    final_t = [[] for _ in range(len(input_val))]
    hop_t = [[] for _ in range(len(input_val))]
    init_p = [[] for _ in range(len(input_val))]
    final_p = [[] for _ in range(len(input_val))]
    hop_p = [[] for _ in range(len(input_val))]

    if (len(input_val)) < cpus:
        cpus = len(input_val)
    sys.stdout.write('CPU: %3d Reading data: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for ntraj, val in enumerate(pool.imap_unordered(read_pyrai2md, input_val)):
        ntraj += 1
        p, natom, nstate, trj_init, trj_final, trj_hop, trj_init_t, trj_final_t, trj_hop_t, trj_init_p, trj_final_p, \
            trj_hop_p = val

        ## send data
        atoms.append(natom)
        states.append(nstate)
        init[p] = trj_init
        final[p] = trj_final
        hop[p] = trj_hop
        init_t[p] = trj_init_t
        final_t[p] = trj_final_t
        hop_t[p] = trj_hop_t
        init_p[p] = trj_init_p
        final_p[p] = trj_final_p
        hop_p[p] = trj_hop_p
        sys.stdout.write(
            'CPU: %3d Reading snapshots: %6.2f%% %d/%d\r' % (cpus, ntraj * 100 / (len(input_val)), ntraj, len(input_val)))

    main_dict = {
        'title': title,
        'natom': max(atoms),
        'nstate': max(states),
        'ntraj': ntraj,
        'init': init,
        'final': final,
        'hop': hop,
        'init_t': init_t,
        'final_t': final_t,
        'hop_t': hop_t,
        'init_p': init_p,
        'final_p': final_p,
        'hop_p': hop_p,
    }

    print('\nSave snapshot data to geom-%s.json\n' % title)
    with open('geom-%s.json' % title, 'w') as out:
        json.dump(main_dict, out)

    return main_dict

def init_prep(read_init, param_list, thrhd):
    with open(read_init, 'r') as data:
        coord = data.read().splitlines()
    input_val = []
    ntraj_list = []
    time_list = []
    gap_list = []
    out_xyz = ''
    i_coord = -1
    natom = int(coord[0].split()[2])
    for n, line in enumerate(coord):  # geom_h[classify_state] is a dict
        if 'Init' in line:
            i_coord += 1
            xyz = coord[n + 1: n + 1 + natom]
            ntraj_list.append(i_coord)
            gap_list.append(0)
            time_list.append(0)
            label = 'traj %s coord 1 state 0' % (i_coord + 1)
            out_xyz += '%s\n%s\n%s\n' % (natom, label, '\n'.join(xyz))
            for i_param, param in enumerate(param_list):
                input_val.append([0, i_coord, i_param, xyz, param, thrhd[i_param], []])

    print('\nRead initial snapshot from %s' % read_init)
    print('\nOverwrite initial snapshot to ini.pmd.xyz')

    with open('ini.pmd.xyz', 'w') as out:
        out.write(out_xyz)

    param = [[[] for _ in param_list] for _ in range(i_coord + 1)]

    return param, input_val, ntraj_list, time_list, gap_list

def special_prep(order, traj_index, geom, step, pot, snapshot_type, label_key, param_list, thrhd, select):
    input_val = []
    ntraj_list = []
    time_list = []
    gap_list = []
    out_xyz = ''
    i_coord = -1
    for traj_idx in traj_index:  # geom_h[classify_state] is a dict

        if len(geom[traj_idx - 1]) == 0:
            continue

        if len(select) > 0:
            if traj_idx not in select:
                continue

        geom_index = -1
        for n, hop_coord in enumerate(geom[traj_idx - 1]):
            label = hop_coord[0]
            if label_key in label:
                geom_index = n

        if geom_index == -1:
            continue

        i_coord += 1
        label = geom[traj_idx - 1][geom_index][0]
        xyz = geom[traj_idx - 1][geom_index][1:]
        energy = pot[traj_idx - 1][geom_index]
        time = step[traj_idx - 1][geom_index]
        gap = ' '.join(['%24.16f' % x for x in energy])
        ntraj_list.append(traj_idx)
        gap_list.append(gap)
        time_list.append(time)
        out_xyz += '%s\n%s\n%s\n' % (len(xyz), label, '\n'.join(xyz))
        for i_param, param in enumerate(param_list):
            input_val.append([order, i_coord, i_param, xyz, param, thrhd[i_param], []])

    print('\nSave %s snapshot to %s.pmd.xyz' % (snapshot_type, snapshot_type))

    with open('%s.pmd.xyz' % snapshot_type, 'w') as out:
        out.write(out_xyz)

    param = [[[] for _ in param_list] for _ in range(i_coord + 1)]

    return param, input_val, ntraj_list, time_list, gap_list

def aligner(var):
    i_geom, coord, ref, core, out = var

    head = coord[0: 2]
    ss = np.array([x.split() for x in coord[2:]])
    atom = ss[:, 0]
    ss = ss[:, 1: 4].astype('float')
    rf = np.array([x.split() for x in ref[2:]])
    rf = rf[:, 1: 4].astype('float')

    if len(core) <= 0:
        core = [x for x in range(len(ss))]
    else:
        core = [x - 1 for x in core]

    if len(out) <= 0:
        out = [x for x in range(len(ss))]
    else:
        out = [x - 1 for x in out]

    p = ss.copy()[core, :]
    q = rf.copy()[core, :]
    pc = p.mean(axis=0)
    qc = q.mean(axis=0)
    p -= pc
    q -= qc
    c = np.dot(np.transpose(p), q)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    if d:  # ensure right-hand system
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)

    new_coord = np.dot(ss[out, :] - pc, u) + qc

    new_coord = '%s\n%s\n' % (len(new_coord), head[1]) + ''.join(
        ['%-5s %18.10f %18.10f %18.10f\n' % (atom[n], x[0], x[1], x[2]) for n, x in enumerate(new_coord)]
    )

    return i_geom, new_coord

def align_mol(title, align_core, output_atom, cpus):
    print('\nAlign snapshots from %s' % title)

    with open(title, 'r') as inxyz:
        xyz = inxyz.read().splitlines()

    natom = int(xyz[0])
    coord_list = []
    for n, line in enumerate(xyz):
        if 'traj' in line:
            coord = xyz[n - 1: n + 1 + natom]
            coord_list.append(coord)

    variables_wrapper = [[n, x, coord_list[0], align_core, output_atom] for n, x in enumerate(coord_list)]
    if (len(variables_wrapper)) < cpus:
        cpus = len(variables_wrapper)

    coord_new = ['' for _ in coord_list]
    sys.stdout.write('CPU: %3d Aligning snapshots: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for p, val in enumerate(pool.imap_unordered(aligner, variables_wrapper)):
        p += 1
        i_geom, new_coord = val
        coord_new[i_geom] = new_coord
        sys.stdout.write(
            'CPU: %3d Aligning snapshots: %6.2f%% %d/%d\r' % (
                cpus, p * 100 / (len(variables_wrapper)), p, len(variables_wrapper))
        )

    print('\nSave aligned snapshot to aligned-%s' % title)
    with open('aligned-%s' % title, 'w') as out:
        out.write(''.join(coord_new))


def RUNspecial(key_dict):
    ## This function compute geometrical parameter for plot
    ## This function unpack main dictionary from raw_data
    ## This function call compute_para for parallelization
    title = key_dict['title']
    cpus = key_dict['cpus']
    select = key_dict['select']
    param = key_dict['param']
    param_list = Paramread(param)
    thrhd = key_dict['thrhd']
    read_init = key_dict['read_init']
    classify = key_dict['classify']
    classify_state = key_dict['classify_state']
    output_atom = key_dict['output_atom']
    align = key_dict['align']
    align_core = key_dict['align_core']
    label_key = '%s to %s' % (classify_state + 1, classify_state)

    if len(thrhd) != len(param_list):
        thrhd = [0 for _ in range(len(param_list))]

    if os.path.exists('geom-%s.json' % title):
        print('\nLoad snapshot data from geom-%s.json' % title)
        with open('geom-%s.json' % title, 'r') as indata:
            geom_data = json.load(indata)
    else:
        print('\nRead geometry data from calculation folders')
        geom_data = RUNreadpmd(key_dict)

    ntraj = geom_data['ntraj']
    coord_i: list = geom_data['init']
    coord_h: list = geom_data['hop']
    coord_f: list = geom_data['final']
    init_t: list = geom_data['init_t']
    final_t: list = geom_data['final_t']
    hop_t: list = geom_data['hop_t']
    init_p: list = geom_data['init_p']
    final_p: list = geom_data['final_p']
    hop_p: list = geom_data['hop_p']

    if len(select) == 0:
        traj_index = [x + 1 for x in range(ntraj)]
    else:
        traj_index = select

    ## compute geometrical parameters for selected trajectories
    ## compute geometrical parameters
    input_val_i = []
    param_i = []
    ntraj_i = []
    time_i = []
    gap_i = []
    if classify == 'all' or classify == 'init':
        if read_init:
            param_i, input_val_i, ntraj_i, time_i, gap_i = init_prep(read_init, param_list, thrhd)
        else:
            param_i, input_val_i, ntraj_i, time_i, gap_i = special_prep(
                0, traj_index, coord_i, init_t, init_p, 'ini', 'state', param_list, thrhd, select
            )
        if align:
            align_mol('ini.pmd.xyz', align_core, output_atom, cpus)

    input_val_f = []
    param_f = []
    ntraj_f = []
    time_f = []
    gap_f = []
    if classify == 'all' or classify == 'final':
        param_f, input_val_f, ntraj_f, time_f, gap_f = special_prep(
            1, traj_index, coord_f, final_t, final_p, 'fin', 'state', param_list, thrhd, select
        )
        if align:
            align_mol('fin.pmd.xyz', align_core, output_atom, cpus)

    input_val_h = []
    param_h = []
    ntraj_h = []
    time_h = []
    gap_h = []
    if classify == 'all' or classify == 'hop':
        param_h, input_val_h, ntraj_h, time_h, gap_h = special_prep(
            2, traj_index, coord_h, hop_t, hop_p, 'hop', label_key, param_list, thrhd, select
        )
        if align:
            align_mol('hop.pmd.xyz', align_core, output_atom, cpus)

    input_val = input_val_i + input_val_f + input_val_h
    param_all = [param_i, param_f, param_h]

    if len(param_list) == 0:
        exit('\nSkip parameter calculations')

    if (len(input_val)) < cpus:
        cpus = len(input_val)

    sys.stdout.write('CPU: %3d Computing parameters: \r' % cpus)
    pool = multiprocessing.Pool(processes=cpus)
    for p, val in enumerate(pool.imap_unordered(compute_para, input_val)):
        p += 1
        i_traj, i_geom, i_param, geom_param, geom_type = val
        param_all[i_traj][i_geom][i_param] = [geom_param, geom_type]
        sys.stdout.write(
            'CPU: %3d Computing parameters: %6.2f%% %d/%d\r' % (cpus, p * 100 / (len(input_val)), p, len(input_val)))

    ## save geometrical parameters
    if len(param_all[0]) > 0:
        classify_output(title, 'init', param_all[0], ntraj_i, 'pmd.ini', time_i, gap_i)

    if len(param_all[1]) > 0:
        classify_output(title, 'final', param_all[1], ntraj_f, 'pmd.fin', time_f, gap_f)

    if len(param_all[2]) > 0:
        classify_output(title, 'hop', param_all[2], ntraj_h, 'pmd.hop', time_h, gap_h)


def main(argv):
    ##  This is the main function.
    ##  It read all options from command line or control file and
    ##  pass them to other module for sampling and generating structures.
    ##  The parser code has clear description for all used variables.
    ##  This function calls getindex to read index

    print('')
    usage = """

    Trajectory analysis script

    Usage:
      python traj_analyzer.py trajectory or
      python traj_analyzer.py for help

    Keywords

      title      name of calculation files
      save_traj  1
      cpus       2
      read_index 1 [single,range,separate range]
      minstep    50  
      maxstep    100
      mode       1
      select      1
      ref_geom   name of the reference xyz file [if has multiple structures the first one is the major product]
      param      a list of geometrical parameter to compute
      thrhd      a list of threshold for classification according to the selected geometrical parameters
      save_data  0

    For more information, please see traj-analyzer-readme.txt

    """

    ## Default settings
    title = None  # Name of calculation files
    cpus = 1  # Number of CPU for analysis
    read_index = '1'  # Index of calculation files to read
    save_traj = 1  # save trajectory data into json
    minstep = 1  # Minimum step per trajectory
    maxstep = 0  # Maximum step per trajectory
    maxdrift = 0.5  # Maximum energy drift to check energy conservation
    opt_mode = 0  # Analysis mode.
    # Skip analysis (0), analyze products and trajectories (1), only analyze products (2), only analyze trajectories (3)
    select = None  # Select trajectories to classify snapshots or fetch trajectory data or compute plot data
    read_init = None  # Read initial condition form a .init or init.xyz file
    classify = None  # Select type of snapshots for classification
    classify_state = 0  # Target state for structure classification
    align = None  # align snapshots
    align_core = []  # define the core for alignment
    output_atom = []  # output the coordinates of the selected atom
    ref_geom = None  # A reference coordinates file that has one or more structures
    param = []  # Geometrical parameters to compute. Options and a file of them are acceptable
    thrhd = []  # Threshold for classification according to the selected geometrical parameters
    save_data = 0  # Save detailed kinetic energy, potential energy, total energy, state population
    prune_type = None  # prune trajectory according to atomic/fragment distances
    prune_index = []  # atom indices to compute geometrical parameters
    prune_thrhd = []  # threshold of geometrical changes to prune trajectory

    prog = 'molcas'  # Output file format

    if len(argv) <= 1:
        exit(usage)

    with open(argv[1]) as inp:
        inputfile = inp.read().splitlines()

    for line in inputfile:
        if len(line.split()) < 2:
            continue
        key = line.split()[0].lower()
        if 'cpus' == key:
            cpus = int(line.split()[1])
        elif 'title' == key:
            title = line.split()[1]
        elif 'read_index' == key:
            read_index = line.split()[1:]
        elif 'save_traj' == key:
            save_traj = int(line.split()[1])
        elif 'minstep' == key:
            minstep = int(line.split()[1])
        elif 'maxstep' == key:
            maxstep = int(line.split()[1])
        elif 'maxdrift' == key:
            maxdrift = float(line.split()[1])
        elif 'mode' == key:
            opt_mode = line.split()[1]
        elif 'read_init' == key:
            read_init = line.split()[1]
        elif 'output_atom' == key:
            output_atom = line.split()[1:]
        elif 'classify' == key:
            classify = line.split()[1]
        elif 'classify_state' == key:
            classify_state = int(line.split()[1])
        elif 'align' == key:
            align = int(line.split()[1])
        elif 'align_core' == key:
            align_core = line.split()[1:]
        elif 'select' == key:
            select = line.split()[1:]
        elif 'ref_geom' == key:
            ref_geom = line.split()[1]
        elif 'param' == key:
            param = line.split()[1:]
        elif 'threshold' == key:
            thrhd = [float(x) for x in line.split()[1:]]
        elif 'prune_type' == key:
            prune_type = line.split()[1].lower()
        elif 'prune_index' == key:
            prune_index = line.split()[1:]
        elif 'prune_thrhd' == key:
            prune_thrhd = [float(x) for x in line.split()[1:]]
        elif 'save_data' == key:
            save_data = int(line.split()[1])
        elif 'prog' == key:
            prog = str(line.split()[1]).lower()

    if title is None:
        print('\n!!! Cannot recognize name of calculations !!!\n')
        print(usage)
        print('\n!!! Cannot recognize name of calculations !!!\n')
        exit()

    print('-------------------------------------')

    if read_index is None:
        print('\n!!! Cannot recognize the trajectory calculation index !!!\n')
        print(usage)
        print('\n!!! Cannot recognize the trajectory calculation index !!!\n')
        exit()

    if os.path.exists('%s' % (read_index[0])):
        print('\nRead index from file: %s' % (read_index[0]))
        opt_index = '%s' % (read_index[0])
        with open(read_index[0], 'r') as para:
            read_files = para.read().split()
    else:
        print('\nRead index from input')
        opt_index = 'input'
        traj_index = getindex(read_index)
        read_files = []
        for i in traj_index:
            read_files.append('%s-%s' % (title, i))

    if len(param) > 0:
        if os.path.exists('%s' % (param[0])):
            print('\nRead geometrical parameters from file: %s' % (param[0]))
            opt_param = '%s' % (param[0])
            with open(param[0], 'r') as para:
                param = para.read().split()
        else:
            print('\nRead geometrical parameters from input')
            opt_param = 'input'
    else:
        print('\nRead geometrical parameters: None')
        opt_param = 'None'

    if maxstep != 0:
        print('\nMaximum step of trajectory: %s' % maxstep)

    if classify is None:
        classify = 'all'

    if classify not in ['all', 'hop', 'final', 'init']:
        classify = 'all'

    if select is not None:
        if os.path.exists('%s' % (select[0])):
            print('\nSelect trajectories from file: %s' % (select[0]))
            opt_select = '%s' % (select[0])
            with open(select[0], 'r') as para:
                select = para.read().split()
        else:
            print('\nSelect trajectories from input')
            opt_select = 'input'
        select = getindex(select)
    else:
        print('\nSelect trajectories: None')
        opt_select = 'None'
        select = []

    if len(output_atom) > 0:
        output_atom = getindex(output_atom)

    if len(align_core) > 0:
        align_core = getindex(align_core)

    if prune_type:
        pindex, pthrhd = set_prune(prune_type, prune_index, prune_thrhd)
    else:
        pindex = []
        pthrhd = []

    log_info = """
      Trajectory Analyzer Log

-------------------------------------
Title:                      %-10s
Mode:                       %-10s
Data from:                  %-10s
Read index:                 %-10s
Compute parameter:          %-10s
Thresholds:                 %-10s
Select trajectories:        %-10s
Classify snapshots:         %-10s
Classify state:             %-10s
Maximum step of trajectory: %-10s
Prune trajectory:           %-10s
Prune parameters:           %-10s
-------------------------------------

    """ % (title, opt_mode, prog, opt_index, opt_param, thrhd, opt_select, classify, classify_state, maxstep,
           prune_type, len(pindex))

    with open('%s.traj.log' % title, 'w') as trajlog:
        trajlog.write(log_info)
    key_dict = {
        'title': title,
        'cpus': cpus,
        'save_traj': save_traj,
        'read_files': read_files,
        'minstep': minstep,
        'maxstep': maxstep,
        'maxdrift': maxdrift,
        'read_init': read_init,
        'classify': classify,
        'classify_state': classify_state,
        'align': align,
        'align_core': align_core,
        'output_atom': output_atom,
        'select': select,
        'ref_geom': ref_geom,
        'param': param,
        'thrhd': thrhd,
        'save_data': save_data,
        'pindex': pindex,
        'pthrhd': pthrhd,
        'prog': prog,
    }

    if opt_mode in ['1', 'diag', 'diagnosis']:
        print('\nRun diagnosis of data')
        RUNdiag(key_dict)
    elif opt_mode in ['2', 'cons', 'conservation']:
        print('\nCheck energy conservation of data')
        RUNcheck(key_dict)
    elif opt_mode in ['3', 'read', 'extract']:
        print('\nExtract trajectory data')
        RUNread(key_dict)
    elif opt_mode in ['4', 'pop', 'population']:
        print('\nCompute state population')
        RUNpop(key_dict)
    elif opt_mode in ['5', 'cat', 'classify']:
        print('\nClassify trajectories')
        RUNclassify(key_dict)
    elif opt_mode in ['6', 'plot', 'compute']:
        print('\nCompute plot data')
        RUNcompute(key_dict)
    elif opt_mode in ['7', 'out', 'fetch']:
        print('\nFetch trajectories')
        RUNfetch(key_dict)
    elif opt_mode in ['8', 'pmd']:
        print('\nFast mode for compute parameters of PyRAI2MD trajectories')
        RUNspecial(key_dict)
    else:
        print('\n!!! Skip analysis !!!\n')
        exit()

    print('-------------------------------------')


if __name__ == '__main__':
    main(sys.argv)
