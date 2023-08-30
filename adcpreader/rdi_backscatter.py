from collections import namedtuple

import numpy as np

from adcpreader import __VERSION__

from adcpreader.coroutine import Coroutine, coroutine


class AcousticAbsorption(object):
    '''
    acousctic absorption according to 
    http://resource.npl.co.uk/acoustics/techguides/seaabsorption/physics.html

    '''

    #def alpha(self,f,T,z):
    #    D=z/1e3 #km
    #    alpha_db_per_km=4.9e-4*(f/1e3)**2*np.exp(-(T/27+D/17))
    #    alpha=alpha_db_per_km*1e-3 # db/m
    #    alpha*=0.23 # ln(10)/10 in neper
    #    return alpha

    def alpha(self, f, T, S, z, pH=7):
        fkhz = f/1e3
        D = z*1e-3 # km
        f1 = 0.78*np.sqrt(S/35)*np.exp(T/26) #khz
        f2 = 42*np.exp(T/17) # khz
        alpha_db_per_km = 0.106*f1*fkhz**2/(f1**2+fkhz**2)*np.exp((pH-8)/0.56)
        alpha_db_per_km += 0.52*(1+T/43)*(S/35)*f2*fkhz**2/(f2**2+fkhz**2)*np.exp(-D/6)
        alpha_db_per_km += 4.9e-4*fkhz**2*np.exp(-(T/27+D/17))
        alpha_db_per_m = alpha_db_per_km*1e-3 # per m
        alpha_np_per_m = alpha_db_per_m * 0.23025
        return alpha_np_per_m

    def __call__(self,f, T, S, z, pH=7):
        return self.alpha(f, T, S, z, pH)

class AcousticCrossSection(Coroutine):
    ''' A class which adds the acoustic cross section area to the ensemble.
        
    A new section 'sigma' is created with the variabels Sigma1..4, and Sigma_AVG.

    db_per_count: about 0.61 for DVL Explorer (see manual)


    '''
    def __init__(self, S=None, k_t=1e-8, N_t=45, db_per_count=[0.61]*4):
        super().__init__()
        self.S = S
        self.__data=[k_t, N_t, db_per_count]
        self.alpha = AcousticAbsorption()
        self.coro_fun = self.coro_add_acoustic_cross_section()
        
    @coroutine
    def coro_add_acoustic_cross_section(self):
        config = None
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                if not config:
                    config = self.get_config(ens)
                self.add_acoustic_cross_section(ens, config)
                self.send(ens)
        self.close_coroutine()

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
    

    def add_acoustic_cross_section(self, ens, config):
        T = ens['variable_leader']['Temp']
        S = self.S or ens['variable_leader']['Salin']
        ens['sigma']=dict()
        ens['sigma']['Sigma_AVG'] = np.zeros_like(ens['echo']['Echo1'])
        for b, db_per_count in enumerate(config.db_per_count):
            EI = ens['echo']['Echo%d'%(b+1)]
            sigma = self.compute_sigma(EI, T, S, db_per_count, config) 
            ens['sigma']['Sigma%d'%(b+1)] = sigma
            ens['sigma']['Sigma_AVG'] += sigma
        ens['sigma']['Sigma_AVG'] /= config.n_beams
    

    def compute_sigma(self, echo_intensity, T, S, db_per_count, config):
        N_t = config.N_t
        k_t = config.k_t
        r = config.r
        f = config.frequency
        
        ei_db = (echo_intensity-N_t) * db_per_count
    
        alpha=self.alpha(f, T, S, r)
        sigma=r**2*np.exp(4*r*alpha)
        sigma*=10**(ei_db/10)
        sigma*=k_t
        return sigma

    
