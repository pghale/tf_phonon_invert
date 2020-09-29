#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:13:34 2020

@author: pg_normal
"""
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
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# Write layer weights
#tf.logging.set_verbosity(tf.logging.ERRORR)

atoms = BULK_pkg('Al', 'fcc', a=4.05)
lattice_vectors = atoms.get_cell()
atoms_per_uc = 1
dofs_per_uc = atoms_per_uc*3
atoms.calc = EMT_pkg()

R1 = lattice_vectors[0][0:3]
R2 = lattice_vectors[1][0:3]
R3 = lattice_vectors[2][0:3]

T = 10
N = pow(3,3) # See forward run, information about 1st BZ

masses = tf.cast(tf.convert_to_tensor(atoms.get_masses()), dtype="float32")
hbar_unit = ase.units._hbar*ase.units.second*ase.units.J
atoms_per_uc = 1
positions = tf.cast(tf.convert_to_tensor(atoms.positions), dtype="float32")

hbar = tf.cast(hbar_unit, dtype="float32")


class ForwardModel(keras.layers.Layer):
    def __init__(self, Nkpoints, atoms_per_uc):
        super(ForwardModel, self).__init__()
        self.Nkpoints = Nkpoints
        self.atoms_per_uc = atoms_per_uc
    
    def build(self, qin_vals):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(Nkpoints, atoms_per_uc*3, atoms_per_uc*3), dtype="float32"), trainable=True, )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(Nkpoints,atoms_per_uc*3, atoms_per_uc*3), dtype="float32"), trainable=True,
            )
        
        # constants for fs-computation
        #define fsvals for aluminum, hardcoded value
        # from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        # accessed Monday, september 14, 2020

        avalues = (6.4202, 1.9002, 1.5936, 1.9646)
        bvalues = (3.0387, 426, 31.5472, 85.0886)
        c=1.1151
        self.a_sigma = tf.constant(0.00001)

        avalues = tf.convert_to_tensor(avalues, dtype="float32")
        bvalues = tf.convert_to_tensor(bvalues, dtype="float32")
        c = tf.convert_to_tensor(c, dtype="float32")

        self.avalues = tf.reshape(avalues, (1,atoms_per_uc, 4))  # same for all values of q/batches, varies w. number of atoms, components
        self.bvalues = tf.reshape(bvalues, (1, atoms_per_uc, 4))
        self.c = tf.reshape(c, (1,atoms_per_uc, 1)) # extra dimensions added for consistent indexing with everything else in code

    def call(self, qin_vals):
        
        # Construct D(k) from self.w:
        Atot = tf.complex(self.w,self.b)
        Atot = tf.add(Atot, tf.linalg.adjoint(Atot))
        Atot = Atot/2.0
        
        [eigvals, eigvecs] = tf.linalg.eigh(Atot)
        min_eig = tf.keras.backend.min(tf.cast(eigvals, dtype="float32"))
        self.add_loss(lambda : -tf.minimum(min_eig,0.00))
        omegas = tf.math.divide(tf.math.sqrt(tf.math.abs(eigvals)), hbar)
        
        # This is  (1/omega)*coth(hbar*omega/(2*kB*T)) = 1/(omega*tanh(hbar*omega/(2*kBT)))s
        omega_coth_prod = tf.linalg.diag(tf.math.divide(1.0, tf.math.multiply(omegas , tf.math.tanh(hbar*omegas/(2*kB*T)) ) ) )
        BZ_products = tf.linalg.matmul(tf.linalg.adjoint(eigvecs), tf.linalg.matmul(tf.cast(omega_coth_prod, dtype="complex64"), eigvecs))
        
        # Check that the product is correct
        ExpectedI = tf.linalg.matmul(eigvecs, tf.linalg.adjoint(eigvecs))
        toMatch = tf.eye(3*atoms_per_uc, batch_shape=[Nkpoints])
        
        # compute sum over BZ
        sum_Bz = tf.reduce_sum(BZ_products, axis=0)
        sum_Bz = tf.expand_dims(sum_Bz, axis=0)
        sum_Bz = tf.broadcast_to(sum_Bz, (Nkpoints, 3*atoms_per_uc, 3*atoms_per_uc))

        
        # qin_vals = [q1,q2,q3] and kpoint, KL point, and index in BZ space
        qin = tf.cast(qin_vals[:,0:3], dtype="complex64")
        kpoints = qin_vals[:,3:6]
        KLpoints = qin_vals[:,6:9]
        
        n_batch = qin_vals.shape[0]

        c = tf.broadcast_to(self.c, (Nkpoints, atoms_per_uc,1))
        avalues = tf.broadcast_to(self.avalues, (Nkpoints, atoms_per_uc, 4))
        bvalues = tf.broadcast_to(self.bvalues, (Nkpoints, atoms_per_uc, 4))

        q2_tensor = tf.math.real(tf.keras.layers.Dot(axes=1)([qin,qin]))
        q2_tensor = tf.expand_dims(q2_tensor,-1)
        q2_tensor = tf.broadcast_to(q2_tensor, (Nkpoints, atoms_per_uc, 4))
        fsvals = tf.math.reduce_sum(c, axis=2) + tf.math.reduce_sum(avalues*tf.exp(-bvalues*q2_tensor), axis=2)
        
        # get diract delta
        k_2 = tf.math.real(tf.math.square(tf.norm(kpoints, ord=2, axis=1)))
        delta = 1.0/(pi*self.a_sigma)*tf.exp(-k_2/pow(self.a_sigma,2))
        delta = tf.expand_dims(delta, -1)
        
        I1 = tf.zeros((n_batch,1),dtype="float32")
        I0 = tf.zeros((n_batch,1),dtype="float32")
        
        for s in range(0, atoms_per_uc):
            #qs = tf.zeros( (3*atoms_per_uc,n_batch), dtype="float32")
            beginIndex = s*3
            endIndex = (s+1)*3
            
            indices = tf.constant([[beginIndex], [beginIndex+1], [beginIndex+2]])
            updates = tf.constant(tf.transpose(qin))
            shape = tf.constant([3*atoms_per_uc, n_batch])
            qs = tf.scatter_nd(indices, updates, shape)
            qs = tf.transpose(qs)
            
            # get Ms:
            coth_q_sumK_s = tf.keras.layers.Dot(axes=1)([tf.math.conj(qs), tf.linalg.matvec(sum_Bz, qs)])
            Ms = (1.0/(4*masses[s]))*(hbar/N)*tf.math.real(coth_q_sumK_s)
            A_s = tf.expand_dims(fsvals[:,s],-1)*tf.exp(-Ms)/tf.math.sqrt(masses[s])
            
            # tau_s
            Tau_s = positions[s,:]
            
            for sp in range(0, atoms_per_uc):
                #qsp = tf.zeros( (n_batch, 3*atoms_per_uc), dtype="float32")
                beginIndex_sp = sp*3
                endIndex_sp = (sp+1)*3
                
                indices_sp = tf.constant([[beginIndex_sp], [beginIndex_sp+1], [beginIndex_sp+2]])
                updates_sp = tf.constant(tf.transpose(qin))
                shape_sp = tf.constant([3*atoms_per_uc, n_batch])
                qsp = tf.scatter_nd(indices_sp, updates_sp, shape_sp)
                qsp = tf.transpose(qsp)
                
                Tau_sp = positions[sp,:]
                
                # get Msp:
                coth_q_sumK_sp = tf.keras.layers.Dot(axes=1)([tf.math.conj(qsp), tf.linalg.matvec(sum_Bz, qsp)])
                Msp = (1.0/(4*masses[sp]))*(hbar/N)*tf.math.real(coth_q_sumK_sp)
                
                # take inner product of latent space with q-vectors
                coth_q_sumK = tf.keras.layers.Dot(axes=1)([tf.math.conj(qs), tf.linalg.matvec(sum_Bz, qsp)])
                coth_q_indv = tf.keras.layers.Dot(axes=1)([tf.math.conj(qs), tf.linalg.matvec(BZ_products, qsp)])
                
                A_sp = tf.expand_dims(fsvals[:,sp],-1)*tf.exp(-Msp)/tf.math.sqrt(masses[sp])
                
                # e^{-iK_L\cdot \tau_{ss'}}
                KL_taus = tf.exp(tf.complex(0.0,-tf.tensordot(KLpoints, Tau_s - Tau_sp, axes=1)))
                KL_taus = tf.expand_dims(KL_taus, -1)
                I1 = I1 + A_s*A_sp*tf.math.real(tf.math.multiply(coth_q_indv,KL_taus))

                # e^{-iq \cdot \tau_{ss'}}
                q_taus = tf.exp(tf.complex(0.0,-tf.tensordot(tf.math.real(qin), Tau_s - Tau_sp, axes=1)))
                q_taus = tf.math.real(tf.expand_dims(q_taus, -1))
                I0 = I0 + tf.math.real( tf.expand_dims(fsvals[:,s]*fsvals[:,sp],-1)*tf.exp(-Msp-Ms)*q_taus)
                

        Itotal = I0*delta + I1
        
        return (Itotal/tf.keras.backend.sum(Itotal, axis=0))
        
         
       
#latent_dim_model = pow(N,3)*pow(3,2)*2

Itotal = NP_pkg.load('Itotal.npy')
qinfo = NP_pkg.load('qinfo.npy')

Itotal = tf.convert_to_tensor(Itotal)
Itotal = tf.cast(Itotal, dtype="float32")
qinfo = tf.convert_to_tensor(qinfo)

Nkpoints = qinfo.shape[0]

optimizer = tf.keras.optimizers.Adadelta()
#SGD(learning_rate=1e-3)

epochs = 10000
tf.keras.backend.set_floatx('float32')

losses = NP_pkg.zeros(epochs)
gradients = NP_pkg.zeros(epochs)
# Iterate over epochs.


forwardLayer = ForwardModel(Nkpoints, atoms_per_uc = 1)
time0 = time.time()
# do stuff


for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    #for step, x_batch_train in enumerate(train_dataset):
    with tf.GradientTape() as tape:

        reconstructed = forwardLayer(qinfo)
        # Compute reconstruction loss
        loss = (tf.norm(reconstructed - Itotal))
        
        loss += (forwardLayer.losses[epoch])  # Add KLD regularization loss
        
        grads = tape.gradient(loss, forwardLayer.trainable_weights)
        optimizer.apply_gradients(zip(grads, forwardLayer.trainable_weights))
        gradients[epoch]= tf.norm(grads)
        losses[epoch] =  loss
        
        print("step %d: mean loss = ", ( loss))
            

elapsed = time.time() - time0
print("Time elapsed [s]:", elapsed)
           
            
        
        




    