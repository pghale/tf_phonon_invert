# tf_phonon_invert
Invert total scattering measurements for phonons

At the moment forward model of phonons D(k) --> I(q) can be computed, but direct inversion has never been attempted. Here, inversion is attempted via backpropagation of one layer.

Software stack: ----------------------------------
        
        conda, pip, tensorflow2.3, (Atomic Simulation Engine) https://wiki.fysik.dtu.dk/ase/ ASE

List of files: -----------------------------------
        
        src files: Aluminum_data_generator.py -- uses ASE module to simulate X-ray scattering, will be replaced by experimental data
                    src_tensorflow.py -- contains the actual forward model of the simulation in tensorflow and optimizes
        
        data files: Itotal.npy -- numpy array of target values
                    qinfo.npy -- momentum space coordinates
        
        Performance files: LossFn_iterations.png -- loss function over 10^4 iteration/epoch
                           gradientnorms_v_iterations.png -- norm of gradient at each iteration

        Misc file :       Al_phonon.png -- Phonon DOS that we expect after convergence.
        
        Theory file:      phonons_inv.pdf



