from collections import deque

import datetime
import numpy as np
from scipy.interpolate import interp1d
import gsw

from rdi import __VERSION__
from rdi import rdi_transforms
from rdi.rdi_reader import get_ensemble_time, unixtime_to_RTC
from rdi.coroutine import coroutine, Coroutine




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
    
    factor_beam1 : factor to correct the EI for beam1
    factor_beam2 : factor to correct the EI for beam2
    factor_beam3 : factor to correct the EI for beam3
    factor_beam4 : factor to correct the EI for beam4


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

class Aggregator(Coroutine):
    '''Class to aggregate a number of ensembles into averages of time,
    roll, pitch, heading, sound speed, salinity temperature, pressure,
    velocity_i and echo_i. Other parameters are taken from the most
    central ensemble.

    Typical use:

    agg = Aggregator(60) # 60 ensembles averaged together


    :
    x.send_to(agg)
    :

    '''
    AVG_PARAMETERS = "Roll Pitch Heading Soundspeed Salin Temp Press Time Timestamp Velocity1 Velocity2 Velocity3 Velocity4 Echo1 Echo2 Echo3 Echo4 BTVel1 BTVel2 BTVel3 BTVel4".split()
    
    def __init__(self, aggregate_size):
        ''' Constructor

        Parameters
        ----------
        aggregate_size: int
            this many ensembles should be aggregated.

        '''
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





