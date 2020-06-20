#!/usr/bin/env python
import os

constants = {

    # physical units
    "kBoltz"        : 1.38064852e-23, # Boltzmann constant, J/K
    "rho"           : 1000.,        # water density, kg/m^3
    "Ndumps"        : 250,           # dumber of dumps

    # numerical approximations
    "Cdt"           : 0.25,         # time step constraint coefficient: sonic and acceleration terms
    "Cdt_visc"      : 0.125,        # time step constraint coefficient: viscous term

    "Gka"           : 4.0e-3,       # numerical rbc global area constraint, J / m^2
    "Gkv"           : 4.0e4,        # numerical rbc global volume constraint, J / m^3

    "a3"            : -2.0,         # higher order stretch coefficients, LWM strain law
    "a4"            :  8.0,         # higher order stretch coefficients, LWM strain law
    "b1"            :  0.7,         # higher order shear coefficients, LWM strain law
    "b2"            :  0.75,        # higher order shear coefficients, LWM strain law

    # simulation units
    "rc"            : 1.0,          # cutoff radius (DPD units, L)
    "mass"          : 1,            # mass of a DPD particle, (DPD units, M)

    "nd"            : 10,           # number density (DPD units, 1/L**3)
    "AIJ"           : 50,           # dimensionless DPD compressibility: aij * rc / kBT
    "kpow"          : 0.125,        # DPD weight function exponent, dimensionless

    "R0"            : 4.0,          # effective cell radius in DPD length units
    "rho0"          : 10,           # density used in viscosity surrogate (DPD units, M/L^3)
    "correctfreq"   : 1000,         # number of timesteps for bounce correction
    "statsfreq"     : 10000,        # number of timesteps for bounce correction
    "safety"        : 1.5,          # safety factor for domain size
    "dtmax"         : 5.e-3        # maximum dt allowed
}

def simulation(u, plasma_par, hemogl_par, rbc_par, mesh_par, sim_par, ply_dir):
    import mirheo as mir
    
    """
        u      : Mirheo object
        plasma : Parameters for outer solvent
        hemogl : Parameters for inner solvent
        rbc    : Parameters for rbc mechanical properties
        mesh   : Parameters for rbc mesh
        sim    : Parameters related to simulation setup
    """

    logfile = open(ply_dir + "config.txt", "a")
    
    if u.isComputeTask()==True:
        logfile.write("\n~~~ Simulation parameters:")
        logfile.write("\n" + str(sim_par))
        logfile.write("\n~~~ RBC mechanical properties:")
        logfile.write("\n" + str(rbc_par))
        logfile.write("\n~~~ Plasma parameters:")
        logfile.write("\n" + str(plasma_par))
        logfile.write("\n~~~ Hemoglobin parameters:")
        logfile.write("\n" + str(hemogl_par))


    # ~~~ Outer solvent: plasma
    plasma_pv   = mir.ParticleVectors.ParticleVector('outer', mass = plasma_par['mass'])
    ic_plasma   = mir.InitialConditions.Uniform(number_density = plasma_par['nd'])
    u.registerParticleVector(pv=plasma_pv, ic=ic_plasma)

    # ~~~ RBC mesh
    mesh_ini = mesh_par['ini']
    mesh_ref = mesh_par['ref']
    mesh_rbc = mir.ParticleVectors.MembraneMesh(mesh_ini.vertices.tolist(), mesh_ref.vertices.tolist(), mesh_ini.faces.tolist())
    rbc_pv   = mir.ParticleVectors.MembraneVector('rbc', mass=mesh_par['mass'], mesh=mesh_rbc)
    ic_rbc   = mir.InitialConditions.Membrane([[sim_par['domain'][0]*0.5, sim_par['domain'][1]*0.5, sim_par['domain'][2]*0.5, 1.0, 0.0, 0.0, 0.0]])
    u.registerParticleVector(pv=rbc_pv, ic=ic_rbc)

    # ~~~ Inner solvent
    checker     = mir.BelongingCheckers.Mesh('checker')
    u.registerObjectBelongingChecker(checker, rbc_pv)
    hemogl_pv = u.applyObjectBelongingChecker(checker=checker, pv=plasma_pv, inside='inner', correct_every=sim_par['correctfreq'])

    # ~~~ Bouncer
    bouncer = mir.Bouncers.Mesh("bounce_rbc", "bounce_maxwell", kBT=0.0)
    u.registerBouncer(bouncer)

    # ~~~ Interactions
    dpd_int = mir.Interactions.Pairwise('dpd', rc=plasma_par['rc'], kind='DPD', a=plasma_par['alpha'], gamma=plasma_par['gamma'], kBT=plasma_par['kbt'], power=plasma_par['kpow'])
    rbc_int  = mir.Interactions.MembraneForces("int_rbc", **rbc_par, stress_free=True)
    u.registerInteraction(dpd_int)
    u.registerInteraction(rbc_int)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    subvv = mir.Integrators.SubStep('subvv', sim_par['substeps'], [rbc_int])
    u.registerIntegrator(subvv)

    if u.isComputeTask():
        dpd_int.setSpecificPair(rbc_pv,    plasma_pv, a=0, gamma=sim_par['gfsi_o'])
        dpd_int.setSpecificPair(rbc_pv,    hemogl_pv, a=0, gamma=sim_par['gfsi_i'])
        dpd_int.setSpecificPair(hemogl_pv, plasma_pv,      gamma=0, kBT=0)
        dpd_int.setSpecificPair(hemogl_pv, hemogl_pv,      gamma=hemogl_par['gamma'])

    u.setInteraction(dpd_int, plasma_pv,   plasma_pv)
    u.setInteraction(dpd_int, hemogl_pv,   plasma_pv)
    u.setInteraction(dpd_int, rbc_pv,      plasma_pv)
    u.setInteraction(dpd_int, hemogl_pv,   hemogl_pv)
    u.setInteraction(dpd_int, rbc_pv,      hemogl_pv)

    # ~~~ Integration
    u.setIntegrator(vv, hemogl_pv)
    u.setIntegrator(vv, plasma_pv)
    u.setIntegrator(subvv, rbc_pv)

    # ~~~ Membrane bounce
    u.setBouncer(bouncer, rbc_pv, hemogl_pv)
    u.setBouncer(bouncer, rbc_pv, plasma_pv)

    # ~~~ Dumps
    logfile.write('Saving results to: ' + ply_dir)
    logfile.write('Current Path: ' + os.getcwd())
    u.registerPlugins(mir.Plugins.createDumpMesh('rbcs', rbc_pv, dump_every=sim_par['dumpfreq'], path = ply_dir))

    logfile.close()
    
    # ~~~ Run
    u.run(sim_par['iend'])

def run_korali( comm_address, gammaC, NDATA, TEND, ini_mesh_fname, ref_mesh_fname, ply_dir, verbose=False, dryrun=False):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constants
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lm      = 10.
    factor  = 0.005

    kT_s    = 0.02
    T_p     = factor * (23.+273.)
    etai_p  = factor * 10.e-3
    etao_p  = etai_p / lm
    mu_p    = factor * 2.5e-6
    Ka_p    = factor * 5.e-6
    kb_p    = factor * 2.e-19
    area_p  = 140.e-12
    volume_p = 100.e-18

    run( comm_address, gammaC, NDATA, TEND, kT_s, T_p, etao_p, etai_p, mu_p, Ka_p, kb_p, area_p, volume_p, verbose, dryrun, ini_mesh_fname, ref_mesh_fname, ply_dir )


def run( comm_address, gammaC, NDATA, TEND, kT_s, T_p, etao_p, etai_p, mu_p, Ka_p, kb_p, area_p, volume_p, verbose, dryrun, ini_mesh_fname, ref_mesh_fname, ply_dir='ply/'):

    import trimesh
    import numpy as np
    from pint import UnitRegistry

    import mirheo as mir
    import dpdParams as D
    import rbcParams as R


    logfile = open(ply_dir + "config.txt", "a")
    
    ureg = UnitRegistry()

    def effective_radius_from_area(A):
        return np.sqrt(A / (4.0 * np.pi))

    @ureg.wraps(None, ureg.dimensionless)
    def to_sim(a):
        return a


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Physical values (_p), in SI units
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A_p             = area_p * ureg.m**2
    V_p             = volume_p * ureg.m**3
    R_p             = effective_radius_from_area(A_p)

    mu_p            = mu_p * ureg.N / ureg.m
    Ka_p            = Ka_p * ureg.N / ureg.m
    kb_p            = kb_p * ureg.J

    kB_p            = constants.get('kBoltz') * ureg.J / ureg.K
    T_p             = T_p * ureg.K
    kT_p            = kB_p * T_p

    Gka_p           = constants.get('Gka') * ureg.J / ureg.m**2
    Gkv_p           = constants.get('Gkv') * ureg.J / ureg.m**3
    a3              = constants.get('a3')
    a4              = constants.get('a4')
    b1              = constants.get('b1')
    b2              = constants.get('b2')

    rho_p           = constants.get('rho')  * ureg.kg / ureg.m**3
    etai_p          = etai_p * ureg.Pa * ureg.s
    etao_p          = etao_p * ureg.Pa * ureg.s

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Non-dimensional numbers:
    #
    # FvK_p   : Foeppl-von-Karman number in healthy cells
    # fKa_p   : Ka / mu
    # lm_p    : ratio between inner and outer viscosity
    # FGKa_p  : dimensionless area constraint coefficient
    # FGKv_p  : dimensionless volume constraint coefficient
    # Ftherm  : ratio between bending modulus and thermal energy
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    FvK_p       = mu_p * R_p*R_p / kb_p
    fKa_p       = Ka_p / mu_p
    FGKa_p      = Gka_p * R_p**2 / kb_p
    FGKv_p      = Gkv_p * R_p**3 / kb_p
    Ftherm      = kb_p / kT_p
    lm_p        = etai_p / etao_p


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simulation (_s)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Simulation length of rbc
    suggested_radius = constants.get('R0')

    # Scale initial mesh to suggested radius
    ini = trimesh.load(ini_mesh_fname, process=False)
    ref = trimesh.load(ref_mesh_fname, process=False)
    original_radius = effective_radius_from_area(ini.area)
    rbc_scaling = suggested_radius / original_radius
    ini.vertices *= rbc_scaling
    ref.vertices *= rbc_scaling

    # set length scale (R_s)
    Nv          = len(ini.vertices)
    Nv_ref      = len(ref.vertices)
    A_s         = ini.area
    V_s         = ini.volume
    R_s         = suggested_radius

    # set mass scale (mass_s)
    nd          = constants.get('nd')
    mass_s      = constants.get('mass')
    rho_s       = mass_s * nd

    # rbc mass, assume membrane is 2D surface
    M_s         = rho_s * A_s
    mmass_s     = M_s / Nv
 
    # set time scale (based on kBT)
    kT_s        = kT_s
    
    # unit scalings
    L_UNIT      = R_p / R_s
    M_UNIT      = rho_p / rho_s * (L_UNIT**3)
    T_UNIT      = np.sqrt( kT_s/kT_p * L_UNIT**2 * M_UNIT )
    F_UNIT      = M_UNIT * L_UNIT / T_UNIT**2
    E_UNIT      = F_UNIT * L_UNIT
    VISC_UNIT   = F_UNIT / L_UNIT**2 * T_UNIT


    # Numerical parameters

    AIJ         = constants.get('AIJ')
    rc          = constants.get('rc')
    rho0_s      = constants.get('rho0')
    aij         = AIJ * kT_s / rc
    cs_s        = D.get_Cs_(aij, nd, mass_s, kT_s)
    kpow        = constants.get('kpow')
    kb_s        = to_sim(Ftherm * kT_s)
    mu_s        = to_sim(FvK_p * kb_s / (R_s**2))
    Ka_s        = to_sim(fKa_p * mu_s)
    kade_s      = 0.  # use Minimum rbc model
    DA0D_s      = 0.
    C0_s        = 0.

    Gka_s       = to_sim(FGKa_p * kb_s / (R_s**2))
    Gkv_s       = to_sim(FGKv_p * kb_s / (R_s**3))

    kT_rbc      = kT_s
    etao_s      = to_sim(etao_p / VISC_UNIT)
    etai_s      = to_sim(lm_p * etao_s)
    nuo_s       = etao_s / rho_s
    nui_s       = etai_s / rho_s
    gij         = D.get_gij(aij, nuo_s*rho0_s)
    gin         = D.get_gij(aij, nui_s*rho0_s)
    gfsi_o      = R.get_gammafsiDPD(nuo_s*rho0_s, kpow, A_s, Nv, nd, rc)
    gfsi_i      = R.get_gammafsiDPD(nui_s*rho0_s, kpow, A_s, Nv, nd, rc)

    gT          = 0.
    gC          = gammaC
    etam_s      = np.sqrt(3)/4. * gC   # approximation, Fedosov2010

    FvK_s       = mu_s * R_s**2 / kb_s
    Ftherm_s    = kb_s / kT_s


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Timestep estimation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~ Solvent timestep:
    #   computed based on the timesteps
    #   from sonic, viscous and acceleration
    #   constraints as defined in Morris1997
    LX_min = min(ini.vertices[:,0]);  LX_max = max(ini.vertices[:,0]);  DX = LX_max - LX_min
    LX_min = min(ref.vertices[:,0]);  LX_max = max(ref.vertices[:,0]);  DX0 = LX_max - LX_min
    vcell_s     = 0.5*(DX-DX0) / to_sim(0.2*ureg.s / T_UNIT)

    h           = D.interparticle_distance(nd)
    Cdt         = constants.get('Cdt')
    Cdt_visc    = constants.get('Cdt_visc')
    dtmax       = constants.get('dtmax')

    dt_sonic    = D.get_dt_sonic(h, cs_s, C=Cdt)
    dt_visc     = D.get_dt_viscous(h, max([etao_s, etai_s]), rho_s, C=Cdt_visc)
    Fdpd        = D.get_total_dpd_force(nd, rc, aij, max([gfsi_o, gfsi_i, gij, gin]), vcell_s, kT_s, min([dt_sonic, dt_visc]))
    dt_acc      = D.get_dt_accel_(h, Fdpd, mass_s, C=Cdt)
    dt_fluid    = min([dtmax, dt_sonic, dt_visc, dt_acc])
    dt          = dt_fluid


    # ~ Membrane substeps:
    #   Computed based on the timesteps
    #   from viscous and acceleration
    #   constraints as defined in Morris1997
    hm = R.intervertex_distance(A_s, Nv)
    Fdpd        = D.get_total_dpd_force(nd, rc, aij, max([gfsi_o, gfsi_i]), vcell_s, kT_s, dt_fluid)
    Fintern     = (mu_s+Ka_s)*hm + (kb_s+kade_s)/hm + Gka_s*hm + Gkv_s*hm**2
    dt_m_acc    = D.get_dt_accel_(hm, Fdpd+Fintern, mmass_s, C=Cdt)
    dt_m_visc   = D.get_dt_viscous(hm, max([etao_s, etai_s, etam_s/hm]), rho_s, C=Cdt_visc)
    dt_rbc      = min([dt_m_visc, dt_m_acc])
    substeps    = min(int(dt_fluid/dt_rbc + 0.5), 100)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Checking/Dumping frequencies
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tend_p = TEND * ureg.s
    
    Ndumps = NDATA
    tdump_s = to_sim(tend_p/(Ndumps-1)/T_UNIT)
    dumpfreq = int(tdump_s/dt+0.5)
    tdump_s = dumpfreq * dt
    iend = (Ndumps-1)*dumpfreq + 1

    correctfreq = constants.get('correctfreq')


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Domain, ranks
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    safety = constants.get('safety')

    LX_min = min(ini.vertices[:,0])
    LX_max = max(ini.vertices[:,0])
    LX = int(safety*(LX_max - LX_min)+0.5)

    LY_min = min(ref.vertices[:,1])
    LY_max = max(ref.vertices[:,1])
    LY = int(safety*(LY_max - LY_min)+0.5)

    LZ_min = min(ini.vertices[:,2])
    LZ_max = max(ini.vertices[:,2])
    LZ = int(safety*(LZ_max - LZ_min)+0.5)

    LX      = LX + np.mod(LX,2)
    LY      = LY + np.mod(LY,2)
    LZ      = LZ + np.mod(LZ,2)

    domain  = (LX, LY, LZ)
    ranks   = (1, 1, 1)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Group parameters
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sim = {
        "domain"        : domain,
        "dumpfreq"      : dumpfreq,
        "correctfreq"   : correctfreq,
        "substeps"      : substeps,
        "iend"          : iend,
        "gfsi_o"        : gfsi_o,
        "gfsi_i"        : gfsi_i,
    }

    plasma  = D.set_dpd_params(aij, gij, kT_s, kpow, rc, nd, mass_s)
    hemogl  = D.set_dpd_params(aij, gin, kT_s, kpow, rc, nd, mass_s)
    rbc     = R.set_rbc_params_ade_lwm(Gka_s, Gkv_s, A_s, V_s, gT, gC, kT_rbc, Ka_s, a3, a4, mu_s, b1, b2, kb_s, C0_s, kade_s, DA0D_s)
    mesh    = R.set_mesh_params(ini, ref, Nv, mmass_s)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start simulation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # chech debug_level (DW)
    u = mir.Mirheo(ranks, domain, dt, debug_level=3, log_filename=ply_dir + 'log', comm_ptr=comm_address, no_splash=True)

    if u.isComputeTask() and verbose:
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\n~~~       Physical Units        ~~~")
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\n>> Non-dimensional quantities:")
        logfile.write("\nFvK  :" + str(FvK_p))
        logfile.write("\nfKa  :" + str(fKa_p))
        logfile.write("\nFGKa_p :" + str(FGKa_p))
        logfile.write("\nFGKv_p :" + str(FGKv_p))
        logfile.write("\nFtherm :" + str(Ftherm))
        logfile.write("\n>> Dimensional quantities:")
        logfile.write("\nmu_p:" + str(mu_p))
        logfile.write("\nKa_p:" + str(Ka_p))
        logfile.write("\nA_p:" + str(A_p))
        logfile.write("\netao_p:" + str(etao_p))
        logfile.write("\netai_p:" + str(etai_p))
        logfile.write("\nRe_p:" + str(vcell_s*L_UNIT/T_UNIT * R_p * rho_p / etao_p))
        logfile.write("\nMa_p:" + str(vcell_s*L_UNIT/T_UNIT / (1500. * ureg.m / ureg.s)))
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\n~~~       Simulation Units      ~~~")
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\n>> Non-dimensional quantities:")
        logfile.write("\nFvK  :" + str(FvK_s))
        logfile.write("\nFtherm:" + str(Ftherm_s))
        logfile.write("\n>> Dimensional quantities:")
        logfile.write("\nR_s:" + str(R_s))
        logfile.write("\nA_s:" + str(A_s))
        logfile.write("\nV_s:" + str(V_s))
        logfile.write("\nkT_s:" + str(kT_s))
        logfile.write("\nmu_s:" + str(mu_s))
        logfile.write("\nKa_s:" + str(Ka_s))
        logfile.write("\nkb_s:" + str(kb_s))
        logfile.write("\nGkv_s:" + str(Gkv_s))
        logfile.write("\nGka_s:" + str(Gka_s))
        logfile.write("\naij:" + str(aij))
        logfile.write("\ngij:" + str(gij))
        logfile.write("\ngin:" + str(gin))
        logfile.write("\ngC:" + str(gC))
        logfile.write("\ngT:" + str(gT))
        logfile.write("\netao_s:" + str(etao_s))
        logfile.write("\netai_s:" + str(etai_s))
        logfile.write("\netam_s:" + str(etam_s))
        logfile.write("\nmmass_s:" + str(mmass_s))
        logfile.write("\nvcell_s:" + str(vcell_s))
        logfile.write("\nNdumps:" + str(Ndumps))
        logfile.write("\ntend:" + str(TEND))
        logfile.write("\ndt:" + str(dt))
        logfile.write("\ndt_visc:" + str(dt_visc))
        logfile.write("\ndt_acc:" + str(dt_acc))
        logfile.write("\ndt_m_visc:" + str(dt_m_visc))
        logfile.write("\ndt_m_acc:" + str(dt_m_acc))
        logfile.write("\nsubsteps:" + str(substeps))
        logfile.write("\ndumpfreq:" + str(dumpfreq))
        logfile.write("\ndumptime:" + str(tdump_s))
        logfile.write("\niend:" + str(iend))
        logfile.write("\ndomain:" + str(domain))
        logfile.write("\nRe_s:" + str(vcell_s * R_s * rho_s / etao_s))
        logfile.write("\nMa_s:" + str(vcell_s / cs_s))
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\n~~~       Unit Transforms       ~~~")
        logfile.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logfile.write("\nL_UNIT:" + str(L_UNIT.to_base_units() ))
        logfile.write("\nT_UNIT:" + str(T_UNIT.to_base_units() ))
        logfile.write("\nM_UNIT:" + str(M_UNIT.to_base_units() ))
        logfile.write("\nChecks:")
        logfile.write("\nR_s/R_p:       " + str((R_s*L_UNIT/R_p)) )
        logfile.write("\nkT_s/kT_p:     " + str((kT_s*E_UNIT/kT_p)) )
        logfile.write("\nkb_s/kb_p:     " + str((kb_s*E_UNIT/kb_p)) )
        logfile.write("\netao_s/etao_p: " + str((etao_s*VISC_UNIT/etao_p)) )

    logfile.close()

    if not dryrun:
        simulation(u, plasma, hemogl, rbc, mesh, sim, ply_dir)
    
    del u
    del plasma
    del hemogl
    del rbc
    del mesh
    del sim
