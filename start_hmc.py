import sys
sys.path.insert(1,'./scripts') 
from run_hmc import run_hmc
import numpy as np

'''

   Use this file to configure and run the HMC.
   
  -> Nrun (int): Number of steps in the chain;
  -> N (int): Number of integration steps in Hamiltonian dynamics;
  -> eps (float): Size of the integration step;
  -> sources_model (string list): List of intrinsic models. Supported: PL, LP, PLC;
  -> init_file (string): Name of initial positions file;
  -> inv_mass (string): Name of inverse mass matrix file;
  -> main_folder (string): 

'''

Nrun = 100000
N = 50       
eps = 1e-3#8e-4  

source_model = ['PL', 'PL', 'PL', 'PL','PL']

init_file = 'initialization2.dat'
inv_mass = 'invmass_matrix3.dat'
#main_folder = 


#-------------------------------------------------------

run_hmc(Nrun, N, eps, source_model, init_file, inv_mass)
