from collections import deque

import datetime
import numpy as np

import gsw


from adcpreader import __VERSION__
from adcpreader import rdi_transforms
from adcpreader.rdi_reader import get_ensemble_time, unixtime_to_RTC
from adcpreader.rdi_transforms import RotationMatrix

from adcpreader.coroutine import coroutine, Coroutine


        
class SpeedOfSoundCorrection(Coroutine):
    Vhor = dict(velocity=['Velocity1', 'Velocity2'],
                bottom_track=['BTVel1', 'BTVel2'])

    V3D = dict(velocity=['Velocity1', 'Velocity2', 'Velocity3'],
               bottom_track=['BTVel1', 'BTVel2', 'BTVel3'])

    def __init__(self, RTC_year_base=2000):
        super().__init__()
        self.RTC_year_base = RTC_year_base

    def get_ensemble_timestamp(self, ens):
        ''' returns timestamp in UTC in unix time (s)'''
        return get_ensemble_time(ens,self.RTC_year_base)


class HorizontalCurrentCorrectionFromSalinityPressure(SpeedOfSoundCorrection):

    def __init__(self, t, SA, P, RTC_year_base=2000):
        '''
        Parameters
        ----------
        t : array
            unix time (s)
        SA : array
             absolute salinity
        P : array
            pressure (dbar)

        RTC_year_base : integer
            reference year to base time on.
            
        '''
        super().__init__(RTC_year_base)
        self.coro_fun = self.coro_current_correction(t, SA, P)
        
    @coroutine
    def coro_current_correction(self, t, SA, P):
        ''' Generator returning ensemble data with corrected HORIZONTAL currents
            given externally measured salinity and pressure

        '''
        coordinate_xfrm_checked = False
        ifun_SA = interp1d(t, SA, assume_sorted=True)
        ifun_P = interp1d(t, P, assume_sorted=True)
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                if not coordinate_xfrm_checked:
                    if ens['fixed_leader']['CoordXfrm']!='Earth':
                        raise ValueError('Expected to have the coordinate transform set to Earth.')
                    coordinate_xfrm_checked = True
                temp = ens['variable_leader']['Temp']
                sound_speed_0 = ens['variable_leader']['Soundspeed']
                tm = self.get_ensemble_timestamp(ens)
                SAi = float(ifun_SA(tm))
                Pi = float(ifun_P(tm))
                sound_speed = gsw.sound_speed_t_exact(SAi, temp, Pi)
                correction_factor = sound_speed/sound_speed_0
                raise ValueError('CHECK HERE IF THE CORRECTION FACTOR MAKES SENSE! in rdi_corrections.py')
                for k, v in self.Vhor.items():
                    for _v in v:
                        ens[k][_v]*=correction_factor
                self.send(ens)
        self.close_coroutine()
        
class CurrentCorrectionFromSalinityPressure(SpeedOfSoundCorrection):

    def __init__(self, t, SA, P, RTC_year_base=2000):
        '''
        Parameters
        ----------
        t : array
            unix time (s)
        SA : array
             absolute salinity
        P : array
            pressure (dbar)

        RTC_year_base : integer
            reference year to base time on.
            
        '''
        super().__init__(RTC_year_base)
        self.coro_fun = self.coro_current_correction(t, SA, P)
        
    @coroutine
    def coro_current_correction(self, t, SA, P):
        ''' Generator returning ensemble data with corrected currents using 
            given externally measured salinity and pressure at the transducer.

        Parameters
        ----------
        t : array
            unix time (s)
        SA : array
             absolute salinity
        P : array
            pressure (dbar)

        '''
        ifun_SA = interp1d(t, SA, assume_sorted=True)
        ifun_P = interp1d(t, P, assume_sorted=True)
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                temp = ens['variable_leader']['Temp']
                sound_speed_0 = ens['variable_leader']['Soundspeed']
                tm = self.get_ensemble_timestamp(ens)
                SAi = float(ifun_SA(tm))
                Pi = float(ifun_P(tm))
                sound_speed = gsw.sound_speed_t_exact(SAi, temp, Pi)
                correction_factor = sound_speed/sound_speed_0
                for k, v in self.Vhor.items():
                    for _v in v:
                        ens[k][_v]*=correction_factor
                self.send(ens)
        self.close_coroutine()
        

class CurrentCorrectionFromSalinity (SpeedOfSoundCorrection):

    def __init__(self, SA, RTC_year_base=2000):
        '''
        Current correction class

        Corrects preset value of absolute salinity at the transducer head.

        Parameters
        ----------
        SA : float
            absolute salinity that is expected at the transducer head.
        RTC_year_base : integer (defaults to 2000)
            reference year for time stamp calculations.
        '''
        super().__init__(RTC_year_base)
        self.coro_fun = self.coro_current_correction(SA)
        
    @coroutine
    def coro_current_correction(self, SA):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                temp = ens['variable_leader']['Temp']
                press = ens['variable_leader']['Press']/1e5*10 # from Pa to dbar
                sound_speed_0 = ens['variable_leader']['Soundspeed']
                sound_speed = gsw.sound_speed_t_exact(SA, temp, press)
                correction_factor = sound_speed/sound_speed_0
                for k, v in self.V3D.items():
                    for _v in v:
                        ens[k][_v]*=correction_factor
                self.send(ens)
        self.close_coroutine()

            
    
class ScaleEchoIntensities(Coroutine):
    ''' Class to scale the echo intensities.

    Parameters
    ----------
    
    factor_beam1 : float
        factor to correct the EI for beam1
    factor_beam2 : float
        factor to correct the EI for beam2
    factor_beam3 : float
        factor to correct the EI for beam3
    factor_beam4 : float
        factor to correct the EI for beam4
    '''
    
    def __init__(self, factor_beam1 = 1.0, factor_beam2 = 1.0, factor_beam3 = 1.0, factor_beam4 = 1.0):
        super().__init__()
        self.factors = [factor_beam1, factor_beam2, factor_beam3, factor_beam4]
        self.coro_fun = self.coro_scale_echos()
        
    @coroutine    
    def coro_scale_echos(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                for i, f in enumerate(self.factors):
                    ens['echo']['Echo%d'%(i+1)]*=f
                ens['echo']['Echo_AVG'] = np.array([ens['echo']['Echo%d'%(i+1)] for i in range(4)]).mean(axis=0)
                self.send(ens)
        self.close_coroutine()


class BinMapping(Coroutine):
    ''' Class to apply bin mapping

    Parameters
    ----------
    pitch_mounting_angle : float
        mounting angle for pitch
    roll_mounting_angle : float
        mounting angle for roll

    
    '''
    
    def __init__(self, pitch_mount_angle=0, roll_mount_angle=0):
        super().__init__()
        self.coro_fun = self.coro_map_bins(pitch_mount_angle,
                                           roll_mount_angle)


    def get_configuration(self, ens, pitch_offset, roll_offset):
        # do the one-off calculations and return a tuple with the results.
        if ens['fixed_leader']['CoordXfrm']!='Beam':
            raise ValueError('In order to apply bin mapping, the coordinate system should be BEAM')
        if ens['fixed_leader']['N_Beams'] != 4:
            raise ValueError('4 Beam data is expected.')
        beam_angle = float(ens['fixed_leader']['Beam_Angle'].split()[0]) * np.pi/180
        r0 = ens['fixed_leader']['FirstBin']
        nbins = ens['fixed_leader']['N_Cells']
        binsize = ens['fixed_leader']['DepthCellSize']
        r = (r0 + np.arange(nbins) * binsize).reshape(1, -1)
        beam_offsets_pitch = np.array([0, 0, beam_angle, -beam_angle]).reshape(-1, 1)
        beam_offsets_roll = np.array([beam_angle, -beam_angle, 0, 0]).reshape(-1, 1)

        centre_bin = nbins//2+1
        n_edges = nbins//2*2+2 # always even
        edges = (np.arange(n_edges)-nbins//2 -0.5) * binsize
        j = np.arange(nbins) # index of velocity readings per beam
        return (pitch_offset, roll_offset, beam_angle, beam_offsets_pitch, beam_offsets_roll,
                r, nbins, centre_bin, edges, j)
        
        
    @coroutine    
    def coro_map_bins(self, pitch_offset, roll_offset):
        try:
            ens = (yield)
        except GeneratorExit:
            self.close_coroutine()
        else:
            config = self.get_configuration(ens, pitch_offset, roll_offset) # < do this only once...
            self.map_bins(ens, config)
            self.send(ens)
            while True:
                try:
                    ens = (yield)
                except GeneratorExit:
                    break
                else:
                    self.map_bins(ens, config)
                    self.send(ens)
            self.close_coroutine()

    def map_bins(self, ens, config):
        ''' Map bins
        
        Parameters
        ----------
        ens : dictionary
            ensemble dictionary
        config : tuple
            configuration and one-off calculations done from the info in the first ens.
        

        Modifies the velocity section only, and remaps velocity readings when necessary.
        '''
        (pitch_offset, roll_offset, beam_angle,
         beam_offsets_pitch, beam_offsets_roll,
         r, nbins, centre_bin, edges, j) = config
        n_beams = 4 # this works for 4 beams only anyway.
        pitch = ens['variable_leader']['Pitch'] * np.pi/180. + pitch_offset
        roll = ens['variable_leader']['Roll'] * np.pi/180. + roll_offset

        CP = np.cos(pitch + beam_offsets_pitch)   
        CR = np.cos(roll + beam_offsets_roll)   
        r33 = CP*CR/np.cos(beam_angle)
        m = r33 @ r

        idx = (j + np.digitize(r-m, edges) - centre_bin).clip(0, nbins-1) # make sure we don't get out of bounds
        
        for i in range(n_beams):
            ens['velocity']['Velocity%d'%(i+1)] = ens['velocity']['Velocity%d'%(i+1)][idx[i]]
        
class Aggregator(Coroutine):
    '''Ensemble aggregator

    Aggregates a number of ensembles into one.

    Parameters
    ----------
    aggregate_size: int
        number of ensembles to aggregate.

    
    The aggregated ensemble contains the average of the following variables:
    
        Roll Pitch Heading Soundspeed Salin Temp Press Time Timestamp
    
        Velocity1 Velocity2 Velocity3 Velocity4
    
        Echo1 Echo2 Echo3 Echo4
    
        BTVel1 BTVel2 BTVel3 BTVel4
    
    
    '''
    AVG_PARAMETERS = "Roll Pitch Heading Soundspeed Salin Temp Press Time Timestamp Velocity1 Velocity2 Velocity3 Velocity4 Echo1 Echo2 Echo3 Echo4 BTVel1 BTVel2 BTVel3 BTVel4".split()
    
    def __init__(self, aggregate_size):
        super().__init__()
        self.aggregate_size = aggregate_size
        self.coro_fun = self.coro_aggregate()

    @coroutine
    def coro_aggregate(self):
        collection = []
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                collection.append(ens)
                if len(collection) == self.aggregate_size:
                    ens_agg = self.aggregate(collection)
                    self.send(ens_agg)
                    collection.clear()
        self.close_coroutine()
        
    def aggregate(self, collection):
        ens = collection[len(collection)//2]
        for s, grp in ens.items():
            for v in grp.keys():
                if v not in Aggregator.AVG_PARAMETERS:
                    continue
                x = [c[s][v] for c in collection]
                # figuring out whether to mean an masked array or a normal array:
                # we don't want to create masked arrays unnecessary...
                if np.any([isinstance(_x, np.ma.MaskedArray) for _x in x]):
                    xm = np.ma.mean(x, axis=0)
                else:
                    xm = np.mean(x, axis=0)
                ens[s][v]=xm
        tm = np.mean([get_ensemble_time(c) for c in collection])
        ens['variable_leader']['RTC'] = unixtime_to_RTC(tm)
        return ens



class AttitudeCorrection(Coroutine):
    def __init__(self):
        super().__init__()
        self.coro_fun = self.coro_attitude_correction()

    @coroutine    
    def coro_attitude_correction(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                heading = ens['variable_leader']['Heading']*np.pi/180.
                pitch = ens['variable_leader']['Pitch']*np.pi/180.
                roll = ens['variable_leader']['Roll']*np.pi/180.
                heading, pitch, roll = self.attitude_correction(heading, pitch, roll)
                ens['variable_leader']['Heading'] = heading*180/np.pi
                ens['variable_leader']['Pitch'] = pitch*180/np.pi
                ens['variable_leader']['Roll'] = roll*180/np.pi
                self.send(ens)
        self.close_coroutine()

    def attitude_correction(self, heading, pitch, roll):
        raise NotImplementedError()

class AttitudeCorrectionTiltCorrection(AttitudeCorrection):
    ''' Corrects the attitude, given a scaling factor and a offset.
    
    Parameters
    ----------
    tilt_correction_factor : float
        factor to multiply pitch and roll with
    pitch_offset : float
        offset in pitch (rad)
    method : string
        which method to use to correct

    
    Two methods are available:
        method == 'simple':
            simply scale pitch and roll and correct for offset
            heading: unaltered
      
        method == 'rotation':
             using rotation matrices to compute the heading when
             applying the corrections in pitch and roll.

    '''
    def __init__(self, tilt_correction_factor, pitch_offset = 0,
                 roll_offset = 0, method = 'rotation'):
        super().__init__()
        self._f = tilt_correction_factor
        self.pitch_offset = pitch_offset
        self.roll_offset = roll_offset
        self.R = rdi_transforms.RotationMatrix()
        self.method = method

    def attitude_correction(self, heading, pitch, roll):
        if self.method == 'simple':
            return self.__attitude_correction_simple(heading, pitch, roll)
        else:
            return self.__attitude_correction(heading, pitch, roll)

    def __attitude_correction_simple(self, heading, pitch, roll):
        pitchc = self._f * pitch + self.pitch_offset
        rollc = self._f * roll + self.roll_offset
        return heading, pitchc, rollc

    
    def __attitude_correction(self, heading, pitch, roll):
        CH = np.cos(heading) 
        CP = np.cos(pitch)   
        CR = np.cos(roll)    
        SH = np.sin(heading) 
        SP = np.sin(pitch)   
        SR = np.sin(roll)

        pitchc = self._f * pitch + self.pitch_offset
        rollc = self._f * roll + self.roll_offset
        
        CPc = np.cos(pitchc)   
        CRc = np.cos(rollc)    
        SPc = np.sin(pitchc)   
        SRc = np.sin(rollc)    

        Rz = np.array([[CH,  SH, 0],
                        [-SH,  CH, 0],
                        [0 ,   0, 1]])
        Ry = np.array([[CP,  0,  -SP],
                        [0 ,  1,  0 ],
                        [SP, 0,  CP]])
        Rx = np.array([[1 ,  0,  0],
                        [0 , CR, -SR],
                        [0 , SR,  CR]])
        Ryc = np.array([[CPc, 0 ,  -SPc],
                        [0 ,  1,  0 ],
                        [SPc, 0,  CPc]])
        Rxc = np.array([[1 ,  0,  0],
                        [0 , CRc, -SRc],
                        [0 , SRc,  CRc]])
        R = Rz @ Ry @ Rx @ Rxc.T @ Ryc.T # use @ matrix operator for multiplication
        headingc = (np.arctan2(R[0,1], R[0,0])+2*np.pi)%(2*np.pi)
        return headingc, pitchc, rollc
        
class AttitudeCorrectionLinear(AttitudeCorrection):
    def __init__(self, a, b):
        super().__init__()
        self.a=a
        self.b=b

    # implement the specific function.
    def attitude_correction(self, heading, pitch, roll):
        return  heading, self.a*pitch+ self.b, roll

class DepthCorrection(Coroutine):

    def __init__(self, t, depth, RTC_year_base=2000):
        super().__init__()
        self.RTC_year_base = RTC_year_base
        self.coro_fun = self.coro_add_depth(t, depth)
        
    @coroutine
    def coro_add_depth(self, t, depth):
        ifun = interp1d(t, depth, assume_sorted=True)
        direction = None
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                tm =  get_ensemble_time(ens,self.RTC_year_base)
                z = float(ifun(tm))
                ens['variable_leader']['XdcrDepth'] = z
                if direction is None:
                    direction = 2*int(ens['fixed_leader']['Xdcr_Facing']=='Down')-1
                    cell_size = ens['fixed_leader']['DepthCellSize']
                    n_cells = ens['fixed_leader']['N_Cells']
                    z0 = ens['fixed_leader']['FirstBin']
                    r = z0 + np.arange(n_cells)*cell_size
                z = direction*r + z
                ens['depth'] = dict(z = z)
                self.send(ens)
        self.close_coroutine()



class KalmanFilter(object):
    ''' A Kalman filter implementation to estimate the water depth.

    The measurement: waterdepth = depth of DVL/ADCP + range till bottom
    '''
    
    def __init__(self, qH, rH):
        '''
        Parameters
        ----------
        qH: float
            uncertainty for water depth model
        rH: float
            uncertainty in range reading
        '''
        self.noise_settings = dict(qH=qH, rH=rH)
        self.t0 = None
        self.F = np.array([[1., 0.],
                           [0., 1.]])
        self.H = np.array([[1., 0.]])
        self.I = np.eye(2)
        self.x_post = None
        
    def initial_step(self,Hp):
        if Hp is None:
            self.x_post = np.array([[100,0]]).T
            self.P_post = np.diag([100, 1])
        else:
            self.x_post = np.array([[Hp,0]]).T
            self.P_post = np.diag([0.2**2, 1])
            

    def update(self, t, Hp):
        ''' Updates Kalman filter

        Parameters
        ----------
        t: float
           time in seconds
        H: float
           measured water_depth (from ADCP to the sea bed + depth reading)
        
        Returns
        -------
        x_post : array of floats
              posterior estimate of the state vector (water_depth, water_depth_rate)
        P_post : 2x2 array of floats
              posterior covariance matrix
        '''
        if self.t0 is None:
            self.initial_step(Hp)
        else:
            dt = t - self.t0
            # set model uncertainties
            qH =self.noise_settings['qH'] # always like this.
            rH =self.noise_settings['rH'] 
            q = qH*np.array([[dt**2/2, dt]]).T
            Q = np.diag(q)
            H = self.H
            # the measurement:
            y = np.array([[Hp]])
            R = np.array([[rH]])
            I = self.I
            P_post = self.P_post
            x_post = self.x_post
            #
            F = self.F
            F[0,1] = dt
            P_pre = F @ P_post @ F.T + Q
            K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + R)
            x_pre = F @ x_post
            if not Hp is None:
                self.x_post = x_pre + K @ ( y - H @ x_pre)
                self.P_post = (I -K @ H) @ P_pre @ (I - K @ H).T + K @ R @ K.T
            else:
                self.x_post = x_pre
                self.P_post = P_pre
        self.t0 = t
        return self.x_post, self.P_post



class CorrectDepthRange(Coroutine):
    ''' Correct Range measurements for pitch and roll 

    '''
    
    def __init__(self, pitch_mount_angle, XdcrDepth_scale_factor = 10, qH=0.0001**2, rH=0.20**2):
        '''
        Parameters
        ----------
        pitch_mount_angle: float
            mounting angle of the DVL in radians
        XdcrDepth_scale_factor: float (default: 10)
            a scale factor to convert depth to meters 
        
        qH: float (default 0.0001**2)
        rH: float (default 0.20**2)
            KalmanFilter settings
        '''
        super().__init__()
        self.XdcrDepth_scale_factor = XdcrDepth_scale_factor
        self.T = RotationMatrix()
        self.nbeams = None
        self.theta = None
        self.kf = KalmanFilter(qH =qH, rH=rH)
        self.pitch_mount_angle = pitch_mount_angle
        self.coro_fun = self.coro_correct_range()
        
    @coroutine
    def coro_correct_range(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                self.correct_ensemble(ens)
                self.send(ens)
        self.close_coroutine()

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
        self.bin_size = ens['fixed_leader']['DepthCellSize']
        r0 = ens['fixed_leader']['FirstBin']
        self.ri = np.arange(n_cells)  * self.bin_size + r0


        
    def correct_ensemble(self, ens):
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
        
        It may be that when this applies the current ensemble has no
        informtion on bottom range. To fill in these data, a Kalman
        filter is used. I am not entirely sure whether this is really
        necessary. The upshot is that we get an estiamte of the water
        depth, which is written to the bottom_track section, as
        variable 'WaterDepth'.

        '''
        if self.nbeams is None:
            self.get_fixed_leader_data(ens)
        # get Range measurements. They are in the ADCP cooridinate, so, divide by cos(theta) to get the range length    
        R = np.diag([ens['bottom_track']['Range%d'%(i+1)] for i in range(self.nbeams)])/np.cos(self.theta) 
        z_dvl = ens['variable_leader']['XdcrDepth'] * self.XdcrDepth_scale_factor # this one runs late normally.
        pitch = ens['variable_leader']['Pitch']  * np.pi/180 
        roll = ens['variable_leader']['Roll']  * np.pi/180
        t = ens['variable_leader']['Timestamp'] 
        T = self.T(0, pitch, roll)
        Bp = T.T @ self.B @ R
        Rz = Bp[2,:] # z component of range measurements.
        # use R measurements that are further than 0.5 m
        Rz_reduced = Rz.compress(Rz>0.5)
        # we require minimum three values, and a std<1 m
        if Rz_reduced.shape[0]>=3 and np.std(Rz_reduced)<1.0:
            y = z_dvl + Rz_reduced.mean()
            x0, P0 = self.kf.update(t, y)
        else: # otherwise get x_post as stored in KF
            y= np.nan
            x0, P0 = self.kf.update(t, y)
        H = x0[0,0] # estimate of water depth.
        h = H - z_dvl
        #r = h*np.cos(self.theta) - self.bin_size # to compare with self.ri
        #for i in range(self.nbeams):
        #    s = 'Velocity%d'%(i+1)
        #    ens['velocity'][s] = self.discard_greater(self.ri, r, ens['velocity'][s])

        bt = ens['bottom_track']
        bt['waterdepth_filtered']=H
        bt['waterdepth'] = y
        bt['altitude'] = h

class AdvanceExternalDataInput(Coroutine):
    ''' Class to advance external data (attitude and depth readings) by a given number of ensembles

    '''
    def __init__(self, n_samples_to_advance):
        super().__init__()
        self.n_advance = n_samples_to_advance
        self.coro_fun = self.advance_ensembles()
        self.backlog = deque(maxlen=self.n_advance)
        
    @coroutine
    def advance_ensembles(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                if len(self.backlog)==self.n_advance:
                    ens_backlog = self.backlog.popleft()
                    vl_backlog = ens_backlog['variable_leader']
                    vl = ens['variable_leader']
                    for k in "XdcrDepth Pitch Roll Heading".split():
                        vl_backlog[k] = vl[k]
                    self.send(ens_backlog)
                # put newled read ensemble in backlog
                self.backlog.append(ens)
        self.close_coroutine()


        
#
#
# I don't think we need this anymore. Not tested.
#
# class ReadAhead(Coroutine):
#     def __init__(self, window_length, centered=False):
#         super().__init__()
#         self.window_length = window_length
#         self.required_length = window_length
#         if centered:
#             self.required_length//=2
#         self.coro_fun = self.coro_readahead()
        
#     @coroutine
#     def coro_readahead(self):
#         self.in_q = deque(maxlen = self.window_length)
#         memo = []
#         n = 0
#         while True:
#             try:
#                 ens = (yield)
#             except GeneratorExit:
#                 break
#             else:
#                 memo.append(v)
#                 r = self.process(v)
#                 if r is None:
#                     n+=1
#                     continue
#                 s = memo.pop(0)
#                 self.send(s)
#         # complete backlog
#         for i in range(n):
#             r = self.process(None)
#             s = memo.pop(0)
#             self.send(s)
#         self.close_coroutine()

#     def process(self, v):
#         if not v is None:
#             self.in_q.append(v)
#         else:
#             try:
#                 self.in_q.popleft()
#             except IndexError:
#                 return None
#         if len(self.in_q)<self.required_length:
#             return None
#         return list(self.in_q)





