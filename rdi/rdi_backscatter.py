from collections import namedtuple

import numpy as np

from rdi import __VERSION__


class AcousticAbsorption(object):
    '''
    acousctic absorption according to 
    http://resource.npl.co.uk/acoustics/techguides/seaabsorption/physics.html

    '''

    def alpha(self,f,T,z):
        D=z/1e3 #km
        alpha_db_per_km=4.9e-4*(f/1e3)**2*np.exp(-(T/27+D/17))
        alpha=alpha_db_per_km*1e-3 # db/m
        alpha*=0.23 # ln(10)/10 in neper
        return alpha

    def __call__(self,f,T,z):
        return self.alpha(f,T,z)

class AcousticCrossSection(object):
    ''' A class which adds the acoustic cross section area to the ensemble.
        
    A new section 'sigma' is created with the variabels Sigma1..4, and Sigma_AVG.

    '''
    def __init__(self, k_t=1e-8, N_t=45, db_per_count=[0.3852]*4):
        self.__data=[k_t, N_t, db_per_count]
        self.alpha = AcousticAbsorption()
        
    def __call__(self, ensembles):
        return self.gen(ensembles)

    def get_config(self, ens):
        fld = ens['fixed_leader']
        # get frequency:
        s = fld['Sys_Freq'] # '600 kHz'
        value, unit = s.split()
        if unit=='Hz':
            factor=1
        elif unit=='kHz':
            factor=1e3
        elif unit=='MHz':
            factor=1e9
        else:
            raise ValueError('Unknown frequency unit')
        frequency = float(value)*factor
        
        # get beam angle
        s = fld['Beam_Angle'] # '20 Degree'
        beam_angle = float(s.split()[0])*np.pi/180.

        #get cell infos
        s = fld['DepthCellSize'] # '2.0'
        bin_size = float(s)
        s = fld['FirstBin'] # '2.9'
        first_bin = float(s)
        s = fld['N_Cells'] # '30'
        n_cells = int(s)
        s = fld['N_Beams'] # '4'
        n_beams = int(s)
        # compute radial distance r
        z = np.arange(n_cells,dtype=float)*bin_size + first_bin
        r = z/np.cos(beam_angle)
        Config = namedtuple('Config', field_names='k_t N_t db_per_count frequency beam_angle bin_size first_bin n_cells n_beams r'.split())
        k_t, N_t, db_per_count = self.__data
        config = Config(k_t, N_t, db_per_count, frequency, beam_angle,
                        bin_size, first_bin, n_cells, n_beams, r)
        return config
    

    def gen(self, ensembles):
        config = None
        for ens in ensembles:
            if not config:
                config = self.get_config(ens)
            ens_mod = self.add_acoustic_cross_section(ens, config)
            yield ens_mod

    def add_acoustic_cross_section(self, ens, config):
        T = ens['variable_leader']['Temp']
        ens['sigma']=dict()
        ens['sigma']['Sigma_AVG'] = np.zeros_like(ens['echo']['Echo1'])
        for b, db_per_count in enumerate(config.db_per_count):
            EI = ens['echo']['Echo%d'%(b+1)]
            sigma = self.compute_sigma(EI, T, db_per_count, config) 
            ens['sigma']['Sigma%d'%(b+1)] = sigma
            ens['sigma']['Sigma_AVG'] += sigma
        ens['sigma']['Sigma_AVG'] /= config.n_beams
        return ens
    
        

    def compute_sigma(self, echo_intensity, T, db_per_count, config):
        N_t = config.N_t
        k_t = config.k_t
        r = config.r
        f = config.frequency
        
        ei_db = (echo_intensity-N_t) * db_per_count
    
        alpha=self.alpha(f, T, r)
        sigma=r**2*np.exp(4*r*alpha)
        sigma*=10**(ei_db/10)
        sigma*=k_t
        return sigma

    
