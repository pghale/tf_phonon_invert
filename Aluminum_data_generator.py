#! python3
# -*- coding: utf-8 -*-
# aluminum forward model of phonons
from math import pi, sqrt
import numpy as NP_pkg
import numpy.linalg as linalg
import scipy as SCIPY_pkg
from ase import Atoms as ATOMS_pkg
import ase
from ase.build import bulk as BULK_pkg
from ase.calculators.emt import EMT as EMT_pkg
from ase.phonons import Phonons as PHONONS_pkg
import matplotlib.pyplot as plt
import scipy.optimize
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

def banana(x):
    y = 100*(pow(x[1] - x[0],2)) + pow(1-x[0],2)
    return y

def getfs_local(q_current, Natoms):
    # for aluminum, hardcoded value
    # For aluminum, the atomic scattering factors are as follows
    # from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    # which in turn relies on iucr-tables
    # accessed Monday, september 14, 2020
    avalues = (6.4202, 1.9002, 1.5936, 1.9646)
    bvalues = (3.0387, 426, 31.5472, 85.0886)
    c=1.1151
    
    fsvals = NP_pkg.zeros((Natoms,1), dtype=float)
    
    for atomcount in range(0, Natoms):
        fsvals[atomcount] = c
        for i in range(0,4):
            fsvals[atomcount] = fsvals[atomcount] + avalues[i]*NP_pkg.exp(-bvalues[i]*NP_pkg.dot(q_current, q_current)/pow(4*pi,2))
    
    return fsvals
    
def getDynamicalMatrix_manual(ph, q_c,N,atoms_supercell_positions,masses):
    # k_in is an 1*3 array 
    # check to confirm that dynamical matrix 
    D_N = ph.D_N;
    D_atm1_atm2 = NP_pkg.zeros( ([3,3]), dtype=complex)
    R_cN = ph.lattice_vectors()
    #print(D_atm1_atm2)
    phase_N = NP_pkg.exp(-2.j * pi * NP_pkg.dot(q_c, R_cN))
    D_q = NP_pkg.sum(phase_N[:, NP_pkg.newaxis, NP_pkg.newaxis]*D_N, axis=0)           
    return D_q
 
def getI0_values(Msvals, fsvals, q_in, Natoms):
    #getDebyeWallerFactor(Dk_BZ, kpoints_all, q_in, T, Natoms)
    I0 = 0.0          
    for s in range(0,Natoms):
        for sp in range(0,Natoms):
            I0 = I0 + fsvals[s]*fsvals[sp]*NP_pkg.exp(-Msvals[s] - Msvals[sp])*NP_pkg.exp(-1j*0)
    return I0


def coth(x):
    y =( NP_pkg.cosh(x))/(NP_pkg.sinh(x))
    return y


def getSumOverModes(Dk, natoms, T):
    sum_f = NP_pkg.zeros( ([3*natoms, 3*natoms]), dtype=float)
    [eigvals, eigvecs] = linalg.eigh(Dk)
    for i in range(0, len(eigvals)):
        if eigvals[i]>1e-10:
            # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
            # from ASE -- phonon documentation
            s = ase.units._hbar * 1e10 / sqrt(ase.units._e * ase.units._amu)
            hbar_omega = eigvals[i]*s
            #print(hbar_omega)
            eigenvector = NP_pkg.reshape(eigvecs[i][:], (3*natoms,1))
            current_contribution = (1/hbar_omega)*coth(hbar_omega/(2*kB*T))*eigenvector*NP_pkg.conjugate(eigenvector.T) 
            sum_f = sum_f + current_contribution
    return sum_f
    

def getSum_over_BZ(Dk_all, kpoints_BZ, natoms, T):
    sum_f = NP_pkg.zeros( ([3*natoms, 3*natoms]), dtype=float)
    hbar_unit = ase.units._hbar*ase.units.second*ase.units.J
    
    for icount in range(0,len(Dk_all)):
        Dk_current = Dk_all[icount][:]
        [eigvals, eigvecs] = linalg.eigh(Dk_current)
        #THZ_to_eV = 0.004136  # (electronvolts)

        #print(eigvecs)
        for i in range(0, len(eigvals)):
            if eigvals[i]>1e-10:
                # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
                # from ASE -- phonon documentation
                s = ase.units._hbar * 1e10 / sqrt(ase.units._e * ase.units._amu)
                hbar_omega = eigvals[i]*s
                omega = hbar_omega/hbar_unit
                #print(hbar_omega)
                eigenvector = NP_pkg.reshape(eigvecs[i][:], (3*natoms,1))
                current_contribution = (1/omega)*coth(hbar_omega/(2*kB*T))*eigenvector*NP_pkg.conjugate(eigenvector.T)
                sum_f = sum_f + current_contribution
            #else:
                #print(i)
                #print(Dk_current)
                #print(kpoints_BZ[icount][:])                   
    return sum_f
            

def getMs_allAtoms(q_in, sum_bz, atoms_per_uc, N, masses):
    NP_pkg.reshape(q_in, (3,1))
    # also units of hbar^2/eV/amu  
    # to Angstrom^2 =  0.00418015929 sq. A
    hbar_unit = ase.units._hbar*ase.units.second*ase.units.J
    q_vector = NP_pkg.zeros( (atoms_per_uc*3,1), dtype=float )
    y = NP_pkg.zeros( (atoms_per_uc,1), dtype=complex)
    for atmcount in range(0, atoms_per_uc):
        q_vector = NP_pkg.zeros( ([atoms_per_uc*3,1]), dtype=float )
        q_vector[atmcount*3+0] = q_in[0]
        q_vector[atmcount*3+1] = q_in[1]
        q_vector[atmcount*3+2] = q_in[2]
        #q_vector = NP_pkg.reshape(q_vector,(3*totalAtoms,1))
        # need to assert shapes to prevent runtime errors at some point 
        y[atmcount] = hbar_unit*NP_pkg.dot( (q_vector.T), NP_pkg.dot(sum_bz,q_vector))/(4*masses[atmcount]*pow(N,3)) #*0.00418015929
    return y
#def getI1_values():
    
def getI1(Ms_local, fs_local, mu_s, Dk, q_current, totalAtoms,T, positions_uc, KL):
    sum_j = getSumOverModes(Dk, atoms_per_uc, T)
    q_s = NP_pkg.zeros( (totalAtoms*3,1), dtype=float )
    q_sp = NP_pkg.zeros( (totalAtoms*3,1), dtype=float )
    I1 = 0
    
    hbar_unit = ase.units._hbar*ase.units.second*ase.units.J
    for s in range(0, totalAtoms):
        q_s = NP_pkg.zeros( ([totalAtoms*3,1]), dtype=float )
        q_s[s*3+0] = q_current[0]
        q_s[s*3+1] = q_current[1]
        q_s[s*3+2] = q_current[2]
        for sp in range(0, totalAtoms):
            q_sp = NP_pkg.zeros( ([totalAtoms*3,1]), dtype=float )
            q_sp[sp*3+0] = q_current[0]
            q_sp[sp*3+1] = q_current[1]
            q_sp[sp*3+2] = q_current[2]
            
            tau_ssp = positions_uc[s][:] - positions_uc[sp][:]
            coef = hbar_unit/2*fs_local[s]*fs_local[sp]/sqrt(mu_s[s]*mu_s[sp])*NP_pkg.exp(1j*NP_pkg.dot(KL,tau_ssp))
            I1 = I1 + coef*NP_pkg.exp(-Ms_local[s] - Ms_local[sp])*NP_pkg.dot( (q_s.T), NP_pkg.dot(sum_j,q_sp))
    return I1

# Setup crystal and EMT calculator
atoms = BULK_pkg('Al', 'fcc', a=4.05)
lattice_vectors = atoms.get_cell()
atoms_per_uc = 1
dofs_per_uc = atoms_per_uc*3
atoms.calc = EMT_pkg()

R1 = lattice_vectors[0][0:3]
R2 = lattice_vectors[1][0:3]
R3 = lattice_vectors[2][0:3]


# Phonon calculator
N = 3
midpoint = N*R1/2 + N*R2/2 + N*R3/2
ph = PHONONS_pkg(atoms, EMT_pkg(), supercell=(N, N, N), delta=0.05)
ph.run()
# Read forces and assemble the dynamical matrix
ph.read(acoustic=True, symmetrize=5)
ph.clean()

path = atoms.cell.bandpath('GXULGK', npoints=100)
bs = ph.get_band_structure(path)

force_constants = ph.get_force_constant();

# get atoms used in the superlattice/phonon calculation used to obtain force constants
atoms_N = ph.atoms*ph.N_c
atoms_supercell_positions = atoms_N.positions
    
# get Monkhorst pack for this supercell:
kpoints_BZ = ase.dft.kpoints.monkhorst_pack([N,N,N])
path = atoms.cell.bandpath(kpoints_BZ)
masses = atoms.get_masses()


Dk_BZ = NP_pkg.zeros( ([len(kpoints_BZ), dofs_per_uc, dofs_per_uc]), dtype=complex);
for k_count in range(0, len(kpoints_BZ)):
    Dk = getDynamicalMatrix_manual(ph, kpoints_BZ[k_count][:], N,atoms_supercell_positions,masses)
    Dk_BZ[k_count][:][:] = Dk


# Now compute a forward model of the I0:
# For each D(k), compute the eigenvalues and eigenvectors
# get sum over brillouin zone required for M_s
Sum_BZ = getSum_over_BZ(Dk_BZ, kpoints_BZ,natoms=1, T=10)
# compute I0
# determine q-values mapped out from points in the BZ 

b123 = atoms.get_reciprocal_cell()

iter_q = 0;
Ms_allAtoms = NP_pkg.zeros( (len(kpoints_BZ)*pow(N,3), atoms_per_uc), dtype=float)
q_all = NP_pkg.zeros( (len(kpoints_BZ)*pow(N,3),3), dtype=float)
q_abs = NP_pkg.zeros( (len(kpoints_BZ)*pow(N,3),1), dtype=float)

iter_m = 0;
G_all = NP_pkg.zeros(  q_all.shape, dtype=float)
kpoints_all = NP_pkg.zeros(q_all.shape, dtype=float)

m_all = NP_pkg.zeros(  (pow(N,3),3), dtype=int)
I1 = NP_pkg.zeros( (len(q_all),1), dtype = float)
I0 = NP_pkg.zeros( (len(q_all),1), dtype = float)
hbar_unit = ase.units._hbar*ase.units.second*ase.units.J

for m1 in range(-1,2):
    for m2 in range(-1,2):
        for m3 in range(-1,2):
            G = m1*b123[:][0] + m2*b123[:][1] + m3*b123[:][2]
            
            m_all[iter_m][:] = NP_pkg.asarray( (m1, m2, m3), dtype=int)
            for kcount in range(0,len(kpoints_BZ)):
                q_current = G + kpoints_BZ[kcount][:]
                q_all[iter_q][:] = q_current
                
                G_all[iter_q][:] = G
                kpoints_all[iter_q][:] = kpoints_BZ[kcount][:]
                
                q_abs[iter_q] = NP_pkg.dot(q_current,q_current)
                Ms_allAtoms[:][iter_q] = getMs_allAtoms(q_current, Sum_BZ, atoms_per_uc, N, masses)
                Ms_local = Ms_allAtoms[:][iter_q]
                fs_local = getfs_local(q_current,Natoms=1)
                # need to multiply by fs/sqrt(mu_s)
                # tau_s = 0 ==> |exp(-iKq. tau_s)|^2 = 1
                # compute fs
                
                mu_s = atoms.get_masses()
                
                Dk = Dk_BZ[kcount][:]
                I1[iter_q] = getI1(Ms_local, fs_local, mu_s, Dk, q_current, atoms_per_uc,T=10, positions_uc=atoms.positions, KL=G)                
                
                if linalg.norm(kpoints_BZ[kcount][:])<1e-6:
                    #print(m1, m2, m3)
                    # Here, pow(n,3) accounts for \sum_m exp(1ik.r) = N \sum_n delta()
                    I0[iter_q] = pow(N,3)*getI0_values(Ms_local, fs_local, q_current, Natoms=1)
                    #print(kpoints_BZ[kcount][:], q_current, I0[iter_q])    
                
                iter_q = iter_q + 1

Itotal = I0 + I1
Itotal = Itotal/sum(Itotal)
#Itotal = NP_pkg.log(Itotal/sum(Itotal))

NP_pkg.save('Itotal', Itotal)
qinfo = NP_pkg.concatenate((q_all, kpoints_all, G_all),axis=1)
NP_pkg.save('qinfo',qinfo)
        

#qin = tf.cast(qin_vals[:,0:3], dtype="complex64")
#        # also broadcast to number of atoms if necessary
        # here only one atom-per-uc
#        kpoints = qin_vals[:,3:6]
#        KLpoints = qin_vals[:,6:9]
#        indexvals = tf.math.round(qin_vals[:,9])
        
#np.save(path/'y', y)

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')#fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(q_all[:,0], q_all[:,1],NP_pkg.log(I0[:,0] + I1[:,0]))
#ax.scatter3D(q_all[:,0],q_all[:,1], q_all[:,2],s=Itotal*10)