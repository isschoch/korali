#!/usr/bin/env python

import numpy as np

# ------ HELPER FUNCTIONS ------ #

def sq(x):
    return np.sqrt(x)

def calc_l0(A,Nv):
    return sq(A*4./(2*Nv-4)/sq(3))


def calc_kp(l0,lm,ks,m):
    return (6*ks*pow(l0,(m+1))*pow(lm,2) - 9*ks*pow(l0,(m+2))*lm + 4*ks*pow(l0,(m+3))) / (4*pow(lm,3)-8*l0*pow(lm,2)+4*pow(l0,2)*lm)


def calc_mu0(x0,l0,ks,kp,m):
    return sq(3)*ks/(4.*l0) * (x0/(2.*pow((1-x0),3)) - 1./(4.*pow((1-x0),2)) + 1./4) + sq(3)*kp*(m+1)/(4.*pow(l0,(m+1)))


# -------- COMPUTE QoIs -------- #

def compute_mu(m, x0, ks, A, Nv):
    l0 = calc_l0(A,Nv)
    lm = l0/x0

    kp = calc_kp(l0, lm, ks, m)
    return calc_mu0(x0, l0, ks, kp, m)

def compute_mu_over_ks(m, x0, A, Nv):
    return compute_mu(m, x0, 1.0, A, Nv)

# from optimal UQ results
def set_rbc_params(mesh, x0, ks):

    m = 2.0
    Nv = len(mesh.vertices)
    A = mesh.area
    V = mesh.volume
    
    # max likelihood estimate from Athena UQ, stretching
    # x0 = 0.48497214
    2576
    # ks = 22.6814565515

    kb = 1.0
    
    prms = {
        "tot_area"   : A,
        "tot_volume" : V,
        "ka_tot"     : 4900.0,
        "kv_tot"     : 7500.0,
        "kBT"        : 0.0,
        "gammaC"    : 52.0,
        "gammaT"    : 0.0,

        "shear_desc": "wlc",
        "ka"        : 5000,
        "x0"        : x0,
        "mpow"      : m,
        "ks"        : ks,

        "bending_desc" : "Kantor",
        "theta"     : 0.0,
        "kb"        : kb
    }
    
    mu = compute_mu(m, x0, ks, A, Nv)
    return prms

# from Fedosov params
def set_rbc_params0(prms, prms_bending, mesh):

    m = 2.0
    Nv = len(mesh.vertices)
    A = mesh.area
    V = mesh.volume
    
    # max likelihood estimate from Athena UQ, stretching
    x0 = 1.0/2.2
    ks = 35.429323407939094

    kb = 27.105156961709344
    
    prms = {
        "tot_area"   : A,
        "tot_volume" : V,
        "ka_tot"     : 4900.0,
        "kv_tot"     : 7500.0,
        "kBT"        : 0.0,
        "gammaC"    : 52.0,
        "gammaT"    : 0.0,

        "shear_desc": "wlc",
        "ka"        : 5000,
        "x0"        : x0,
        "mpow"      : m,
        "ks"        : ks,

        "bending_desc" : "Kantor",
        "theta"     : 0.0,
        "kb"        : kb
    }
    mu = compute_mu(m, x0, ks, A, Nv)
    
    return prms

def print_rbc_params(p):
    print("A = {}".format(p.totArea))
    print("V = {}".format(p.totVolume))
    print("x0 = {}".format(p.x0))
    print("kb = {}".format(pb.kb))
    print("ks = {}".format(p.ks))
    print("m = {}".format(p.mpow))
    
    
