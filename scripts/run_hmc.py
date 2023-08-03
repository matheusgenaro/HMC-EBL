import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGInterp
import os
from datetime import datetime
import shutil

#-----------------------------------------------------------------
# Run Hamiltonian Monte Carlo

def run_hmc(Nrun, N, eps, source_model, init_file, inv_mass):
    
    main_folder = os.getcwd()
    
    # Read data -- May need change if different formatting
    E, flux, sigma, z, name, source_file = read_data(main_folder)
    N_sources = len(source_file)
    
    # Print on terminal
    print_simulation_hmc(Nrun, N, eps, source_model, source_file)
        
    labels = set_labels(source_model)
    Ndim = len(labels)
    
    # Optical depth grid
    f_star, f_pah, f_sg, f_lg = read_grid(main_folder+'/ebl_files/')

    tauStar = []
    tauSG = []
    tauLG = []
    tauPAH = []

    # Interpolate on data
    for i in range(N_sources):
        points = []
        for j in range(len(E[i])):
            points.append([np.log10(z[i]),np.log10(E[i][j])])
        tauStar.append(f_star(points))
        tauPAH.append(f_pah(points))
        tauSG.append(f_sg(points))
        tauLG.append(f_lg(points))
    
    # Initialization
    norm = np.loadtxt('./normalization.dat')
    for i in range(len(flux)):
        for j in range(len(flux[i])):
            flux[i][j] = flux[i][j]*norm[i]
            sigma[i][j] = sigma[i][j]*norm[i]
            
    inv_mass_ = np.loadtxt(inv_mass)
    mass = np.linalg.inv(inv_mass_)
    p0 = generate_momenta(Ndim, mass)
    q0 = np.loadtxt(init_file)

    # Chain file
    filename = main_folder+'/chains/chain_run.dat'
    
    chain = open(filename,'w')
    for j in range(Ndim):
        chain.write(str(q0[j])+'  ')
    chain.write('\n') 
    
    ##
    q, p = forest_ruth_save(q0, p0, eps, N, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, inv_mass_)
    ##
    af = 0
    for i in range(Nrun):
        p0_copy = np.copy(p0)
        q0_copy = np.copy(q0)
        q, p = forest_ruth(q0, p0, eps, N, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, inv_mass_)
        q0, p0, af = accept_criteria(q0_copy,p0_copy,q,p,inv_mass_,E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, af)
        for j in range(Ndim):
            chain.write(str(q0[j])+' ')
        chain.write('\n') 
        p0 = generate_momenta(Ndim, mass)               
    chain.close()
    
    # Compute acceptance fraction
    af = 1 - af/Nrun
    print(af)
    file_chain = change_name_hmc(af, N_sources, Nrun, filename, main_folder+'/chains')

    print_results_hmc(file_chain, af, source_model, name, source_file, Nrun, main_folder)

    return
    
#--------------------------------------------------------------------
# Log-prior

def lnprior_hmc(q):
    if (1 - q[0] - q[1] < 0) or (q[0] < 0) or (q[1] < 0):
        return -np.inf
    else:
        return 0.
#--------------------------------------------------------------------
# Log-likelihood 

def lnlike_hmc(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model):
    E0 = 1. # TeV
    frac_pah = q[0]
    frac_sg = q[1]
    frac_lg = 1 - frac_pah - frac_sg
    
    chi2_total = 0
    j = 0
    for i in range(len(E)):
        # Calculating the optical depth
        tau = tauStar[i] + tauPAH[i]*frac_pah + tauSG[i]*frac_sg + tauLG[i]*frac_lg

        # Intrinsic model
        if source_model[i] == 'PL':
            N0 = q[2+i+j]
            gamma = q[2+i+j+1]
            flux_intr = N0*(E[i]/E0)**(-gamma)
            j += 1
        elif source_model[i] == 'LP':
            N0 = q[2+i+j]
            a = q[2+i+j+1]
            b = q[2+i+j+2]
            flux_intr = N0*(E[i]/E0)**(-a-b*np.log10(E[i]/E0))
            j += 2
        else:
            N0 = q[2+i+j]
            gamma = q[2+i+j+1]
            Ecut = q[2+i+j+2]/(1+z[i])
            flux_intr = N0*((E[i]/E0)**(-gamma))*np.exp(-(E[i]/Ecut))
            j += 2

        flux_mod = np.exp(-tau)*flux_intr
    
        # Chi-square
        chi2 = np.sum(((flux_mod - flux[i])/sigma[i])**2)

        if chi2 is np.nan:
           print('Warning: not a number in chi2')
           chi2 = np.inf

        chi2_total += chi2

    return -0.5*chi2_total

#--------------------------------------------------------------------
# Potential energy

def Potential(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model):
    return -lnlike_hmc(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model) - lnprior_hmc(q)


#--------------------------------------------------------------------
# Kinect energy

def Kinect(p, inv_mass):
    return 0.5*np.matmul(p, np.matmul(inv_mass,p))

#--------------------------------------------------------------------
# Partial derivative of kinect energy

def partial_K(p, inv_mass):
    return np.matmul(inv_mass, p)

#--------------------------------------------------------------------
# Potential energy partial derivative with respect to dust fractions.
# tauDust is the optical depth of the corresponding fraction.

def partial_U_frac(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, tauDust):
    E0 = 1. # TeV
    frac_pah = q[0]
    frac_sg = q[1]
    frac_lg = 1 - frac_pah - frac_sg
    
    grad = 0
    j = 0
    for i in range(len(E)):
        # Calculating the optical depth
        tau = tauStar[i] + tauPAH[i]*frac_pah + tauSG[i]*frac_sg + tauLG[i]*frac_lg

        # Intrinsic model
        if source_model[i] == 'PL':
            N0 = q[2+i+j]
            gamma = q[2+i+j+1]
            flux_intr = N0*np.power(E[i]/E0,-gamma,dtype=np.longdouble)
            #if np.any(np.isinf(flux_intr)):
            #    print(i)
            #    print(gamma)
            #    print(N0)
            #    print(q)
#            flux_intr = N0*(E[i]/E0)**(-gamma)
            j += 1
        elif source_model[i] == 'LP':
            N0 = q[2+i+j]
            a = q[2+i+j+1]
            b = q[2+i+j+2]
            flux_intr = N0*(E[i]/E0)**(-a-b*np.log10(E[i]/E0))
            j += 2
        elif source_model[i] == 'PLC':
            N0 = q[2+i+j]
            gamma = q[2+i+j+1]
            Ecut = q[2+i+j+2]/(1+z[i])
            flux_intr = N0*((E[i]/E0)**(-gamma))*np.exp(-(E[i]/Ecut))
            j += 2

        flux_mod = np.exp(-tau)*flux_intr
        value = np.sum(((flux[i] - flux_mod)/sigma[i])*(flux_mod/sigma[i])*(tauDust[i]-tauLG[i]))

        grad += value

    return grad


#--------------------------------------------------------------------
# Potential energy partial derivative with respect to PL gamma
# Receives E,flux,sigma, etc. from single source and
# params are intrinsic parameters.

def partial_U_PL(frac_pah, frac_sg, params, E, flux, sigma, tauStar, tauSG, tauLG, tauPAH):
    E0 = 1.
    frac_lg = 1 - frac_pah - frac_sg
    tau = tauStar + tauPAH*frac_pah + tauSG*frac_sg + tauLG*frac_lg

    N0 = params[0]
    gamma = params[1]
    f_mod = np.power(E/E0,-gamma,dtype=np.longdouble)*np.exp(-tau, dtype = np.longdouble)
    #f_mod = (E/E0)**(-gamma)*np.exp(-tau)
    
    grad_N0 = np.sum(((N0*f_mod-flux)/sigma)*(f_mod/sigma))
    grad_gamma = np.sum(((N0*f_mod - flux)/sigma)*(-np.log(E/E0)*N0*f_mod/sigma))

    return grad_N0, grad_gamma
    

#--------------------------------------------------------------------
# Potential energy partial derivative with respect to LP params.
# Receives E,flux,sigma, etc. from single source and
# params are intrinsic parameters.

def partial_U_LP(frac_pah, frac_sg, params, E, flux, sigma, tauStar, tauSG, tauLG, tauPAH):
    E0 = 1.
    frac_lg = 1 - frac_pah - frac_sg
    tau = tauStar + tauPAH*frac_pah + tauSG*frac_sg + tauLG*frac_lg

    N0 = params[0]
    a = params[1]
    b = params[2]
    f_mod = (E/E0)**(-a-b*np.log10(E/E0))*np.exp(-tau)
    
    grad_N0 = np.sum((N0*f_mod-flux)*f_mod/sigma**2)
    grad_a = np.sum((N0*f_mod - flux)*(-np.log(E/E0)*N0*f_mod)/sigma**2)
    grad_b = np.sum((N0*f_mod - flux)*(-N0*f_mod*np.log(10)*(np.log10(E/E0))**2)/sigma**2)

    return grad_N0, grad_a, grad_b
    

#--------------------------------------------------------------------
# Potential energy partial derivative with respect to PLC params.
# Receives E,flux,sigma, etc. from single source and
# params are intrinsic parameters.

def partial_U_PLC(frac_pah, frac_sg, params, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH):
    E0 = 1.
    frac_lg = 1 - frac_pah - frac_sg
    tau = tauStar + tauPAH*frac_pah + tauSG*frac_sg + tauLG*frac_lg

    N0 = params[0]
    gamma = params[1]
    Ecut = params[2]/(1+z)
    f_mod = (E/E0)**(-gamma)*np.exp(-E/Ecut, dtype=np.longdouble)*np.exp(-tau)
#    f_mod = (E/E0)**(-gamma)*np.exp(-E/Ecut)*np.exp(-tau)

    grad_N0 = np.sum((N0*f_mod-flux)*f_mod/sigma**2)  
    grad_gamma = np.sum((N0*f_mod - flux)*(-np.log(E/E0)*N0*f_mod)/sigma**2)
    grad_ecut = np.sum((N0*f_mod - flux)*(N0*f_mod*E/Ecut**2)/sigma**2)

    return grad_N0, grad_gamma, grad_ecut

#--------------------------------------------------------------------
# Evolution of coordinate variables for leapfrog algorithm
def evol_coord(q, p, eps, inv_mass,theta):
    q += theta*eps*partial_K(p,inv_mass)
    return 

#--------------------------------------------------------------------
# Evolution of momentum variables for symplectic integrator

def evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, theta):
    # EBL dust fractions
    p[0] += -theta*eps*partial_U_frac(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, tauPAH)
    p[1] += -theta*eps*partial_U_frac(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, tauSG)
    k = 0
    for j in range(len(E)):
    # Intrinsic parameters
        if source_model[j] == 'PL':
            grad_N0, grad_gamma = partial_U_PL(q[0], q[1], q[2+j+k:4+j+k], E[j], flux[j], sigma[j], tauStar[j], tauSG[j], tauLG[j], tauPAH[j])
            p[j+2+k] += -theta*eps*grad_N0
            p[j+2+k+1] += -theta*eps*grad_gamma
            k += 1
        elif source_model[j] == 'LP':
            grad_N0, grad_a, grad_b = partial_U_LP(q[0], q[1], q[2+j+k:5+j+k], E[j], flux[j], sigma[j], tauStar[j], tauSG[j], tauLG[j], tauPAH[j])
            p[j+2+k] += -theta*eps*grad_N0
            p[j+2+k+1] += -theta*eps*grad_a
            p[j+2+k+2] += -theta*eps*grad_b
            k += 2
        else:
            grad_N0, grad_gamma, grad_ecut = partial_U_PLC(q[0], q[1], q[2+j+k:5+j+k], E[j], z[j], flux[j], sigma[j], tauStar[j], tauSG[j], tauLG[j], tauPAH[j])
            p[j+2+k] += -theta*eps*grad_N0
            p[j+2+k+1] += -theta*eps*grad_gamma
            p[j+2+k+2] += -theta*eps*grad_ecut
            k += 2
    return 
 
#--------------------------------------------------------------------
# Leapfrog algorithm for Hamiltonian evolution.

def leapfrog(q, p, eps, N, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, inv_mass):
    for i in range(N):
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 0.5)
        evol_coord(q, p, eps, inv_mass, 1.)
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 0.5)

    return q, p

#--------------------------------------------------------------------
# Forest-Ruth algorithm for Hamiltonian evolution.

def forest_ruth(q, p, eps, N, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, inv_mass):
    for i in range(N):
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 1/(2-2**(1/3)))
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))

        evol_coord(q, p, eps, inv_mass, -0.5*(2**(1/3)/(2-2**(1/3))))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, -(2**(1/3)/(2-2**(1/3))))
        evol_coord(q, p, eps, inv_mass, -0.5*(2**(1/3)/(2-2**(1/3))))
        
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 1/(2-2**(1/3)))
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))

    return q, p

#--------------------------------------------------------------------
# Forest-Ruth algorithm for Hamiltonian evolution.

def forest_ruth_save(q, p, eps, N, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, inv_mass):
    q_file = open('forestruth_q_steps_eps'+str(eps)+'_N'+str(N)+'.txt','w')
    p_file = open('forestruth_p_steps_eps'+str(eps)+'_N'+str(N)+'.txt','w')
    for i in range(N):
        for j in range(len(q)):
            q_file.write(str(q[j])+'  ')
            p_file.write(str(p[j])+'  ')
        q_file.write('\n')
        p_file.write('\n')
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 1/(2-2**(1/3)))
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))

        evol_coord(q, p, eps, inv_mass, -0.5*(2**(1/3)/(2-2**(1/3))))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, -(2**(1/3)/(2-2**(1/3))))
        evol_coord(q, p, eps, inv_mass, -0.5*(2**(1/3)/(2-2**(1/3))))
        
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))
        evol_momenta(q, p, eps, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, 1/(2-2**(1/3)))
        evol_coord(q, p, eps, inv_mass, 0.5/(2-2**(1/3)))
    q_file.close()
    p_file.close()
    return q, p


#--------------------------------------------------------------------
# Acceptance criteria
def accept_criteria(q0, p0, q, p, inv_mass, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model, af):
    if np.any(q[26:-1] < 0):
        af += 1
        return q0, p0, af
    k_new = Kinect(p,inv_mass)
    p_new = Potential(q, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model)
    k_old = Kinect(p0,inv_mass)
    p_old = Potential(q0, E, z, flux, sigma, tauStar, tauSG, tauLG, tauPAH, source_model)
    new = k_new + p_new           
    old = k_old + p_old
    #print('New')
    #print(k_new)
    #print(p_new)
    #print('Old')
    #print(k_old)
    #print(p_old)
    prob = np.exp(-(new - old))
    #print(prob)
    alfa = np.random.uniform()
    if alfa < prob:
        return q, p, af
    else:
        af += 1
        return q0, p0, af
        
#--------------------------------------------------------------------
# Sample momenta from gaussian distribution
def generate_momenta(n, inv_mass):
    return np.random.multivariate_normal(np.zeros(n),inv_mass)

#--------------------------------------------------------------------
# Initialize generalized coordinates
def initialize_q(Ndim, ranges):
    q0 = np.array([])
    proposal = np.full(Ndim, 0.6)
    while (proposal[0] + proposal[1] > 1):
        proposal = np.random.rand(Ndim) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    q0 = np.append(q0, proposal)

    return q0

#=======================================================================
# Change the file name after a simulation
def change_name_hmc(af, Nsources, Nrun, file_sampler, directory):

    af_s = str('{0:.3f}'.format(af))

    file_chain = directory+'/'+str(Nsources)+'Sources'+'_Nrun'+str(Nrun)+'_af'+af_s+'.dat'

    j = 1
    m = 0
    while os.path.isfile(file_chain):
        file_chain = file_chain[:-(4+m)]+'_'+str(j)+'.dat'
        j+=1 
        m = 2

    os.rename(file_sampler, file_chain)

    return file_chain

#=======================================================================
# Print the results and save them in a .txt file

def print_results_hmc(file_chain, af, source_model, name, source_file, Nrun, main_folder):

   # Current time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

   # Open the sampler
    print('Chain saved as '+file_chain)

    af_s = str('{0:.3f}'.format(af))

   # File to save results
    n = len(main_folder)+len('/chains/')
    results_file = open(main_folder+'/results/'+file_chain[n+1:-4]+'.txt','w')
    results_file.write(file_chain+'\n') 
    
   # Write results
    results_file.write('\n'+'#--------- Simulation Info ---------#'+'\n')
    results_file.write('Number of sources: '+str(len(name))+'\n')
    results_file.write('Sources: '+str(name[0])+'\n')
    i = 1
    while i < len(name):
        results_file.write(str(name[i])+'\n')
        i+=1
    results_file.write('Data File: '+str(source_file[0])+'\n')
    i = 1
    while i < len(source_file):
        results_file.write(str(source_file[i])+'\n')
        i+=1
    results_file.write('Model: ')
    for i in range(len(source_model)):
        results_file.write(source_model[i]+' ')
    results_file.write('\n')
    results_file.write('Number of Steps: '+str(Nrun)+'\n')
    results_file.write('Acceptance Fraction: '+af_s+'\n')
    results_file.write('Simulation date/time: '+dt_string+'\n')
        
    results_file.close()
    
    print('Results saved as '+'results/'+file_chain[n+1:-4]+'.txt')

   # Copy File to recent result
    shutil.copyfile(main_folder+'/results/'+file_chain[n+1:-4]+'.txt', r'recent_result.txt')

    return

#=======================================================================
# Create file with median values
def print_values_hmc(samples, labels, model_folder, chain_name, Nburn, thin):
    results_file = open('results/'+chain_name+'_Nburn'+str(Nburn)+'_thin'+str(thin)+'_params_values.txt','w')
    results_file.write('VARIABLE  MEDIAN  SIGMA_PLUS  SIGMA_MINUS\n')
    for i in range(len(labels)):
        q_16, q_50, q_84 = quantile(samples[:,i], [0.16, 0.5, 0.84])
        q_m, q_p = q_50-q_16, q_84-q_50
        results_file.write(labels[i]+' '+str(q_50)+' '+str(q_p)+' '+str(q_m)+'\n') 

    # LG
    q_16, q_50, q_84 = quantile(1-samples[:,0]-samples[:,1], [0.16, 0.5, 0.84])
    q_m, q_p = q_50-q_16, q_84-q_50
    results_file.write(r'$f_{LG}$'+' '+str(q_50)+' '+str(q_p)+' '+str(q_m)+'\n') 

    # Samples
    Nsamples = len(samples[:,0])
    results_file.write("Number of samples: "+str(Nsamples)+'\n')

    results_file.close()
    print('Results saved in results/'+chain_name+'_Nburn'+str(Nburn)+'_thin'+str(thin)+'_params_values.txt') 

    return 

#=======================================================================
# Read Data

def read_data(main_folder):
    sources = np.loadtxt(main_folder+'/sources_list.txt', dtype='str')
    
    E_source = []
    flux_source = []
    sigma = []
    z_source = []
    name = []
    
    for i in range(len(sources)):
        spec = np.loadtxt(main_folder+'/'+sources[i], skiprows=1)
        if spec.ndim == 1:
            E_source.append(spec[0])
            flux_source.append(spec[3])
            sigma.append(spec[4])
        else:
            E_source.append(spec[:,0])
            flux_source.append(spec[:,3])
            sigma.append(spec[:,4])

        with open(main_folder+'/'+sources[i], 'r') as g:
            lines = g.readlines()

        source_info = lines[0].split(', ')
        name.append(source_info[0][3:])
        z_source.append(float(source_info[2][2:]))
        g.close()

    return E_source, flux_source, sigma, z_source, name, sources 
    
#=======================================================================
# Print simulation
def print_simulation_hmc(Nrun, N, eps, source_model, source_file):
    print('------ Starting HMC Simulation ------')
    print('Number of steps: '+str(Nrun))
    print('Forest-Ruth integration steps: '+str(N))
    print('Precision: '+str(eps))
    print('Source models: '+str(source_model))
    print('Source files: '+str(source_file))
    return

#=======================================================================
# Set parameter labels
def set_labels(source_model):

    labels = [r'$f_{PAH}$',r'$f_{SG}$']

    for i in range(len(source_model)):
        if source_model[i] == 'PL':
            labels.append(r'$N_0$')
            labels.append(r'$\Gamma$')

        elif source_model[i] == 'LP':
            labels.append(r'$N_0$')
            labels.append(r'$a$')  
            labels.append(r'$b$')            

        elif source_model[i] == 'PLC':
            labels.append(r'$N_0$')
            labels.append(r'$\Gamma$')  
            labels.append(r'$E_{cut}$') 

    return labels

#=======================================================================
# Read optical depth grid and do interpolation

def read_grid(grid_folder):
    E_grid = np.loadtxt(grid_folder+'ENERGY_TAU.dat')
    z_grid = np.loadtxt(grid_folder+'REDSHIFT_TAU.dat')

    tauStar = np.loadtxt(grid_folder+'TAU_STAR_HUBBLE70.dat')
    tauSG = np.loadtxt(grid_folder+'TAU_SG_HUBBLE70.dat')
    tauLG = np.loadtxt(grid_folder+'TAU_LG_HUBBLE70.dat')
    tauPAH = np.loadtxt(grid_folder+'TAU_PAH_HUBBLE70.dat')

    tauStar_grid = np.reshape(tauStar, (len(z_grid),len(E_grid)))
    tauSG_grid = np.reshape(tauSG, (len(z_grid),len(E_grid)))
    tauLG_grid = np.reshape(tauLG, (len(z_grid),len(E_grid)))
    tauPAH_grid = np.reshape(tauPAH, (len(z_grid),len(E_grid)))

    # Interpolação prévia
    f_star = RGInterp((np.log10(z_grid),np.log10(E_grid)),tauStar_grid)
    f_pah = RGInterp((np.log10(z_grid),np.log10(E_grid)),tauPAH_grid)
    f_sg = RGInterp((np.log10(z_grid),np.log10(E_grid)),tauSG_grid)
    f_lg = RGInterp((np.log10(z_grid),np.log10(E_grid)),tauLG_grid)

    return f_star, f_pah, f_sg, f_lg


