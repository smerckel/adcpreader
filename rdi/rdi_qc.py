import numpy as np

from rdi import __VERSION__

from rdi.coroutine import coroutine, Coroutine
from rdi.rdi_transforms import RotationMatrix

# default setting for if true, all ensembles that have all data
# blanked out because of some quality check will silently be dropped.
DROP_MASKED_ENSEMBLES_BY_DEFAULT = False
    

class QualityControl(Coroutine):
    ''' Quality Control base class

        Implements conditions to make arrays masked arrays,
    
        scalars that don't pass the condition are set to nan.
    '''
    def __init__(self, drop_masked_ensembles=None):
        super().__init__()
        self.conditions = list()
        self.operations = {">":self.discard_greater,
                           ">=":self.discard_greater_equal,
                           "<":self.discard_less,
                           "<=":self.discard_less_equal,
                           "||>":self.discard_abs_greater,
                           "||>=":self.discard_abs_greater_equal,
                           "||<":self.discard_abs_less,
                           "||<=":self.discard_abs_less_equal}
        if drop_masked_ensembles is None:
            self.drop_masked_ensembles = DROP_MASKED_ENSEMBLES_BY_DEFAULT
        else:
            self.drop_masked_ensembles = drop_masked_ensembles
        self.coro_fun = self.coro_check_ensembles()
        
    @coroutine
    def coro_check_ensembles(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                keep_ensemble = self.check_ensemble(ens)
                if keep_ensemble or not self.drop_masked_ensembles:
                    self.send(ens)
                else:
                    continue # ensemble is dropped
        self.close_coroutine()


        
    def check_ensemble(self, ens):
        ''' check an ensemble. Should be subclassed.'''
        raise NotImplementedError

    def discard_greater(self, v, value, vp=None):
        ''' discard values v that are greater than value '''
        condition = v>value
        return self.apply_condition(v, condition, vp)

    def discard_greater_equal(self, v, value, vp=None):
        ''' discard values v that are greater or equal than value '''
        condition = v>=value
        return self.apply_condition(v, condition, vp)

    def discard_less(self, v, value, vp=None):
        ''' discard values v that are less than value '''
        condition = v<value
        return self.apply_condition(v, condition, vp)

    def discard_less_equal(self, v, value, vp=None):
        ''' discard values v that are less or equal than value '''
        condition = v<value
        return self.apply_condition(v, condition, vp)

    def discard_abs_greater(self, v, value, vp=None):
        ''' discard values v that are absolute greater than value '''
        condition = np.abs(v)>value
        return self.apply_condition(v, condition, vp)

    def discard_abs_greater_equal(self, v, value, vp=None):
        ''' discard values v that are absolute greater or equal than value '''
        condition = np.abs(v)>=value
        return self.apply_condition(v, condition, vp)

    def discard_abs_less(self, v, value, vp=None):
        ''' discard values v that are absolute less than value '''
        condition = np.abs(v)<value
        return self.apply_condition(v, condition, vp)

    def discard_abs_less_equal(self, v, value, vp=None):
        ''' discard values v that are absolute less or equal than value '''
        condition = np.abs(v)<value
        return self.apply_condition(v, condition, vp)

    def apply_condition(self, v, condition, vp=None):
        if not vp is None:
            _v = vp
        else:
            _v = v
        try:
            _v.mask |= condition 
        except AttributeError as e:
            if e.args[0] == "'numpy.ndarray' object has no attribute 'mask'":
                _v = np.ma.masked_array(_v, condition)
            else:
                if condition:
                    _v = np.nan
        return _v
    
                
class ValueLimit(QualityControl):
    ''' Qualtiy Control class to mask values that are exceeding some limit.'''
    VECTORS = 'velocity correlation echo percent_good'.split()
    SCALARS = ['bottom_track']
    
    def __init__(self, drop_masked_ensembles=None):
        super().__init__(drop_masked_ensembles)
        
    def set_discard_condition(self, section, parameter, operator, value):
        ''' Set a condition to discard readings.

        Parameters
        ----------
        section : string
            section name of the data block. Example: velocity
        parameter : string
            name of parameter in this section. Example Velocity1
        operator : string
            comparison operator. Example: ">" or "||>"
        value : float
            the value to compare with.
        '''
        self.conditions.append((section, parameter, operator, value))

    def check_ensemble(self, ens):
        keep_ensemble = True
        mask_ensemble = False
        
        for section, parameter, operator, value in self.conditions:
            if section != 'variable_leader':
                continue
            v = ens[section][parameter]
            f = self.operations[operator]
            _v = f(v, value)
            # we don't put nans in the variable leader. If the check causes a positive, mask the
            # the variables in the sections SCALARS and VECTORS (see above).
            # ens[section][parameter] = _v
            if np.isnan(_v):
                mask_ensemble = True
                keep_ensemble = False
        if mask_ensemble:
            for section in ValueLimit.VECTORS:
                if section not in ens.keys():
                    continue
                for k, v in ens[section].items():
                    ens[section][k]=np.ma.masked_array(v, True)
            for section in ValueLimit.SCALARS:
                if section not in ens.keys():
                    continue
                for k, v in ens[section].items():
                    ens[section][k]=np.nan
        else:
            for section, parameter, operator, value in self.conditions:
                if section == 'variable_leader':
                    continue # already done
                if section not in ens.keys():
                    continue
                v = ens[section][parameter]
                f = self.operations[operator]
                _v = f(v, value)
                ens[section][parameter] = _v
                if np.isscalar(_v): # if parameter is scalar and nan, drop the ens.
                    if np.isnan(_v):
                        keep_ensemble = False
                else:
                    if np.all(_v.mask): # if all values are masked, drop it too.
                        keep_ensemble = False
        return keep_ensemble
            


class SNRLimit(QualityControl):
    def __init__(self, SNR_limit = 10, noise_floor_db = 26.1):
        super().__init__()
        self.SNR_limit = SNR_limit
        self.noise_floor_db = noise_floor_db

    def SNR(self, echointensity):
        return 10**((echointensity-self.noise_floor_db)/10)
    
    def check_ensemble(self, ens):
        nbeams = ens['fixed_leader']['N_Beams']
        s = ["Echo%d"%(i+1) for i in range(nbeams)]
        SNR = [self.SNR(ens['echo'][_s])  for _s in s]
        for i,snr in enumerate(SNR):
            if i:
                condition|= snr < self.SNR_limit
            else:
                condition = snr < self.SNR_limit
        for i in range(nbeams):
            s="Velocity%d"%(i+1)
            ens['velocity'][s] = self.apply_condition(ens['velocity'][s], condition)
        return True # always return the ensemble


class AcousticAmplitudeLimit(QualityControl):
    def __init__(self, amplitude_limit = 75):
        super().__init__()
        self.amplitude_limit = amplitude_limit

    def SNR(self, echointensity):
        return 10**((echointensity-self.noise_floor_db)/10)
    
    def check_ensemble(self, ens):
        nbeams = ens['fixed_leader']['N_Beams']
        s = ["Echo%d"%(i+1) for i in range(nbeams)]
        amplitudes = [ens['echo'][_s]  for _s in s]
        for i,amplitude in enumerate(amplitudes):
            if i:
                condition|= amplitude > self.amplitude_limit
            else:
                condition = amplitude > self.amplitude_limit
        for i in range(nbeams):
            s="Velocity%d"%(i+1)
            ens['velocity'][s] = self.apply_condition(ens['velocity'][s], condition)
        return True # always return the ensemble

class KalmanFilter(object):
    ''' A Kalman filter implementation to estimate the water depth.

    The measurements are: depth of DVL/ADCP
                          range till bottom
    '''
    
    def __init__(self, qw, qH, rz, rH, pitch_min = 5, pitch_max = 45):
        '''
        Parameters
        ----------
        qw: float
            uncertainty for velocity model
        qH: float
            uncertainty for water depth model
        rz: float
            uncertainty in depth reading
        rH: float
            uncertainty in range reading
        pitch_min: float
            min value of (absolute) pitch when depth readings are considered
            unit: degree
        pitch_max: float
            max value of (absolute) pitch for which depth readings are considered
            unit: degree
        '''
        self.noise_settings = dict(qw=qw, qH=qH, rz=rz, rH=rH)
        self.t0 = None
        self.F = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])
        self.Hzh = np.array([[1., 0., 0.],
                             [0., 0., 1.]])
        self.pitch_min_rad = pitch_min * np.pi/180.
        self.pitch_max_rad = pitch_max * np.pi/180
        # if we measure the positino only, then R == r
        self.I = np.eye(3)
        
    def initial_step(self, t, z):
        r = self.noise_settings['rz']
        self.x_post = np.array([[z, 0,100]]).T
        self.P_post = np.diag([r, 1, 100])
        
    def update(self, t, z, pitch, h):
        ''' Updates Kalman filter

        Parameters
        ----------
        t: float
           time in seconds
        z: float
           depth reading of ADCP/DVL
        pitch: float
            measured pitch
        h: float
           measured depth range (from ADCP to the sea bed)
        
        Returns
        -------
        x_post : array of floats
              posterior estimate of the state vector (depth, U_glider, water_depth)
        P_post : 3x3 array of floats
              posterior covariance matrix
        '''
        if self.t0 is None:
            self.initial_step(t, z)
        else:
            sn = np.sin(pitch)
            pitch_abs = np.abs(pitch)
            dt = t - self.t0
            # set model uncertainties
            qH =self.noise_settings['qH'] # always like this.
            qw = self.noise_settings['qw']
            q = np.array([[dt**2/2 * np.abs(sn)*qw, dt/2*qw, 2*qH]]).T
            Q = np.diag(q)
            if pitch_abs>self.pitch_min_rad and pitch_abs<self.pitch_max_rad and z>5:
                # the measurement uncertainties
                rz = self.noise_settings['rz'] # depth reading is always like this
            else:
                rz=1000
            if h>1: # valid reading
                rh = self.noise_settings['rH']
            else:
                rh=1000 # if h==0 we didn't get a reading. Set its uncertainty to very high
            H = self.Hzh
            # the measurement:
            y = np.array([[z, z+h]]).T
            R = np.diag([rz, rh])
            
            I = self.I
            P_post = self.P_post
            x_post = self.x_post
            #
            F = self.F
            F[0,1] = dt *sn
            P_pre = F @ P_post @ F.T + Q
            K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)
            x_pre = F @ x_post
            self.x_post = x_pre + K @ ( y - H @ x_pre)
            self.P_post = (I -K @ H) @ P_pre @ (I - K @ H).T + K @ R @ K.T
        self.t0 = t
        return self.x_post, self.P_post



class VelocityRangeLimit(QualityControl):
    ''' Quality check for measured velocity profiles.

    Any velocity readings appear to originate from the sea bottom and deeper are
    masked.
    Additionally, an estimate of the water depth is added to the ensemble.
    '''
    
    def __init__(self, pitch_mount_angle, XdcrDepth_scale_factor = 10,qw=0.005**2, qH=0.001**2, rz = 0.15**2, rH=0.20**2):
        '''
        Parameters
        ----------
        pitch_mount_angle: float
            mounting angle of the DVL in degrees
        XdcrDepth_scale_factor: float (default: 10)
            a scale factor to convert depth to meters 
        
        qw: float (default 0.005**2)
        qH: float (default 0.001**2)
        rz: float (default 0.15**2)
        rH: flaot (default 0.20**2)
            KalmanFilter settings
        '''
        super().__init__()
        self.XdcrDepth_scale_factor = XdcrDepth_scale_factor
        self.T = RotationMatrix()
        self.nbeams = None
        self.theta = None
        self.kf = KalmanFilter(qw=qw, qH =qH, rz=rz, rH=rH)
        self.pitch_mount_angle = pitch_mount_angle*np.pi/180.
        
    def get_fixed_leader_data(self, ens):
        if ens['fixed_leader']['Xdcr_Facing']!='Down':
            raise ValueError('The VelocityRangeLimit operator requires the DVL/ADCP to be downward looking')
            # this is not strictly necessary, but if uipward looking the B array is different, and possibly some
            # other things that I have not thought about, as it is not relevant for glider work.
        # gets data from the fixed leader and other one off computations
        nbeams = ens['fixed_leader']['N_Beams']
        theta_deg, unit = ens['fixed_leader']['Beam_Angle'].split()
        theta = np.pi/180.*float(theta_deg)
        self.nbeams = nbeams
        self.theta = theta
        self.B = np.array([[-np.sin(theta), np.sin(theta), 0. , 0. ],
                           [0., 0., np.sin(theta), -np.sin(theta) ],
                           [np.cos(theta), np.cos(theta), np.cos(theta), np.cos(theta)],
                           [0 , 0, 0, 0]])
        n_cells = ens['fixed_leader']['N_Cells']
        bin_size = ens['fixed_leader']['DepthCellSize']
        r0 = ens['fixed_leader']['FirstBin']
        self.ri = np.arange(n_cells)  * bin_size + r0
        
    def check_ensemble(self, ens):
        '''check ensemble for passing/nonpassing of velocity profile data
        
        Parameters
        ----------
        ens : dictionary
            ensemble dictionary

        Returns
        -------
        True  
            

        The strategy followed here in is the following. The DVL
        records the distance to the seabed as seen by each of its four
        beams. Sometimes velotity readings are registered that seem to
        come from within the seabed due to reflections I suppose. We
        compute the distance from the transducer for the beam that has
        observed the smallest range, as this one, and all the others
        will contaminate the velocity reading. This cell and all
        further cells are masked. 
        
            It may be that when this applies the current ensemble has
        no informtion on bottom range. To fill in these data, a Kalman
        filter is used. I am not entirely sure whether this is really
        necessary. The upshot is that we get an estiamte of the water
        depth, which is written to the bottom_track section, as
        variable 'WaterDepth'.
        '''
        if self.nbeams is None:
            self.get_fixed_leader_data(ens)

        ranges = np.array([ens['bottom_track']['Range%d'%(i+1)] for i in range(self.nbeams)])
        z_dvl = ens['variable_leader']['XdcrDepth'] * self.XdcrDepth_scale_factor
        pitch = ens['variable_leader']['Pitch']  * np.pi/180
        roll = ens['variable_leader']['Roll']  * np.pi/180
        T = self.T(0, pitch, roll)
        Bp = T @ self.B
        h = self.correct_depth_reading(ranges, pitch, roll)
        hm = np.median(h, axis=0)
        ens['bottom_track']['WaterDepth'] = hm + z_dvl
        R = hm / Bp[2]
        Rmin = np.min(R)
        r = Rmin/np.cos(self.theta) #length of shortest beam
        for i in range(self.nbeams):
            s = 'Velocity%d'%(i+1)
            ens['velocity'][s] = self.discard_greater(self.ri, r, ens['velocity'][s])
        return True # always return the ensemble


    def correct_depth_reading(self, h, pitch, roll):
        # see http://psc.apl.uw.edu/HLD/Bstrait/WoodgateandHolroyd2011_BTrangeCorrection.pdf
        sn_theta = np.sin(self.theta)
        p = pitch + self.pitch_mount_angle
        r = roll
        term2 = np.cos(self.theta) * np.sqrt(1 - np.sin(r)**2 - np.sin(p)**2)
        cosai = np.array([np.sin(r), -np.sin(r), -np.sin(p), +np.sin(p)])*sn_theta + term2
        f = cosai/np.cos(self.theta) 
        return  f * h

class Counter(Coroutine):
    ''' An ensemble counter class.

    This class merely counts the number of ensembles that pass through the pipeline at this stage.
    This implies that no ensemble is modified.
    
    The number of ensembles counted are stored in the property counts.

    An instance of this class can be placed at more than one position within the pipeline. The counts 
    property is a list that reflects the positions where the counter is placed.

    '''
    
    def __init__(self, verbose=False):
        super().__init__()
        self.counts = []
        self.coro_fun = self.coro_counter(verbose)

    @coroutine
    def coro_counter(self, verbose=False):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                n = ens['variable_leader']['Ensnum']
                self.counts.append(n)
                if verbose:
                    print("Ensemble : {:4d}".format(n))
                self.send(ens)
        self.close_coroutine()
