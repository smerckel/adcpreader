from collections import deque

import datetime
import numpy as np
from scipy.interpolate import interp1d
import gsw

from rdi import __VERSION__
from rdi import rdi_transforms
from rdi.rdi_reader import get_ensemble_time, unixtime_to_RTC
from rdi import rdi_hardiron

class SpeedOfSoundCorrection(object):
    Vhor = dict(velocity=['Velocity1', 'Velocity2'],
                bottom_track=['BTVel1', 'BTVel2'])

    V3D = dict(velocity=['Velocity1', 'Velocity2', 'Velocity3'],
               bottom_track=['BTVel1', 'BTVel2', 'BTVel3'])

    def __init__(self, RTC_year_base=2000):
        self.RTC_year_base = RTC_year_base

    def get_ensemble_timestamp(self, ens):
        ''' returns timestamp in UTC in unix time (s)'''
        return get_ensemble_time(ens,self.RTC_year_base)
    
    def horizontal_current_from_salinity_pressure(self, ensembles, t, SA, P):
        ''' Generator returning ensemble data with corrected HORIZONTAL currents
            given externally measured salinity and pressure
        
        Parameters
        ----------
        t : array
            unix time (s)
        SA : array
             absolute salinity
        P : array
            pressure (dbar)

        Returns
        -------
        Generator
        '''
        coordinate_xfrm_checked = False
        ifun_SA = interp1d(t, SA, assume_sorted=True)
        ifun_P = interp1d(t, P, assume_sorted=True)
        for ens in ensembles:
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
            yield ens
            
    def current_correction_at_transducer_from_salinity_pressure(self, ensembles, t, SA, P):
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
        
        Returns
        -------
        Generator

        '''
        ifun_SA = interp1d(t, SA, assume_sorted=True)
        ifun_P = interp1d(t, P, assume_sorted=True)
        for ens in ensembles:
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
            yield ens

    def current_correction_at_transducer_from_salinity(self, ensembles, absolute_salinity):
        ''' Current correction generator.
        
        Generator returning ensemble data with corrected currents using 
        a prescribed salinity

        Parameters
        ----------
        ensembles:  list 
            list of ensembles or ensemble generator
        absolute_salinity : float
            absolute salinity that is expected at the transducer head.
        '''

        for ens in ensembles:
            temp = ens['variable_leader']['Temp']
            press = ens['variable_leader']['Press']/1e5*10 # from Pa to dbar
            sound_speed_0 = ens['variable_leader']['Soundspeed']
            sound_speed = gsw.sound_speed_t_exact(absolute_salinity, temp, press)
            correction_factor = sound_speed/sound_speed_0
            for k, v in self.V3D.items():
                for _v in v:
                    ens[k][_v]*=correction_factor
            yield ens

class MotionBias(object):
    V3D = dict(velocity=['Velocity1', 'Velocity2', 'Velocity3'],
               bottom_track=['BTVel1', 'BTVel2', 'BTVel3'])

    def __init__(self, v_bias):
        self.v_bias = v_bias

    def __call__(self, ensembles):
        for ens in ensembles:
            # require "ship" coordinates.
            if ens['fixed_leader']['CoordXfrm'] != 'Ship':
                raise ValueError('Motion bias correction requires at this stages "ship" coordinates.')
            ens['velocity']['Velocity2']+=self.v_bias
            yield ens
            
    
class ScaleEchoIntensities(object):
    ''' Class to scale the echo intensities.

    Parameters
    ----------
    
    factor_beam1 : factor to correct the EI for beam1
    factor_beam2 : factor to correct the EI for beam2
    factor_beam3 : factor to correct the EI for beam3
    factor_beam4 : factor to correct the EI for beam4


    '''
    
    def __init__(self, factor_beam1 = 1.0, factor_beam2 = 1.0, factor_beam3 = 1.0, factor_beam4 = 1.0):
        self.factors = [factor_beam1, factor_beam2, factor_beam3, factor_beam4]
        
    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def gen(self, ensembles):
        for ens in ensembles:
            for i, f in enumerate(self.factors):
                ens['echo']['Echo%d'%(i+1)]*=f
            ens['echo']['Echo_AVG'] = np.array([ens['echo']['Echo%d'%(i+1)] for i in range(4)]).mean(axis=0)
            yield ens

class Aggregator(object):
    '''Class to aggregate a number of ensembles into averages of time,
    roll, pitch, heading, sound speed, salinity temperature, pressure,
    velocity_i and echo_i. Other parameters are taken from the most
    central ensemble.

    Typical use:

    agg = Aggregator(60) # 60 ensembles averaged together


    :
    
    ens = agg(ens) # returns new generator ens.

    :

    '''
    AVG_PARAMETERS = "Roll Pitch Heading Soundspeed Salin Temp Press Time Timestamp Velocity1 Velocity2 Velocity3 Velocity4 Echo1 Echo2 Echo3 Echo4".split()
    
    def __init__(self, aggregate_size):
        ''' Constructor

        Parameters
        ----------
        aggregate_size: int
            this many ensembles should be aggregated.

        '''
        self.aggregate_size = aggregate_size
        
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
        collection.clear()
        return ens


    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def gen(self, ensembles):
        collection = []
        for k, ens in enumerate(ensembles):
            collection.append(ens)
            if ((k+1)%self.aggregate_size) == 0:
                ens_agg = self.aggregate(collection)
                yield ens_agg

class Timeshifter(object):
    RTC_year_base=2000
    def __init__(self, time_offset):
        self.time_offset = time_offset

    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def gen(self, ensembles):
        for ens in ensembles:
            try:
                ens['variable_leader']['Timestamp']+=self.time_offset
            except KeyError:
                ''' returns timestamp in UTC in unix time (s)'''
                tm = get_ensemble_time(ens,self.RTC_year_base) + self.time_offset
                ens['variable_leader']['Timestamp'] = tm
            yield ens
        

class AttitudeCorrection(object):
    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def gen(self, ensembles):
        for ens in ensembles:
            heading = ens['variable_leader']['Heading']*np.pi/180.
            pitch = ens['variable_leader']['Pitch']*np.pi/180.
            roll = ens['variable_leader']['Roll']*np.pi/180.
            heading, pitch, roll = self.attitude_correction(heading, pitch, roll)
            ens['variable_leader']['Heading'] = heading*180/np.pi
            ens['variable_leader']['Pitch'] = pitch*180/np.pi
            ens['variable_leader']['Roll'] = roll*180/np.pi
            yield ens
            
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
        which methd to use to correct
    '''
    # Two methods are available:
    #     method == 'simple':
    #         simply scale pitch and roll and correct for offset
    #         heading: unaltered
        
    #      method == 'rotation':
    #          using rotation matrices to compute the heading when
    #          applying the corrections in pitch and roll.
    # '''
    def __init__(self, tilt_correction_factor, pitch_offset = 0,
                 roll_offset = 0, method = 'simple'):
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

        Rz = np.matrix([[CH,  SH, 0],
                        [-SH,  CH, 0],
                        [0 ,   0, 1]])
        Ry = np.matrix([[CP,  0,  -SP],
                        [0 ,  1,  0 ],
                        [SP, 0,  CP]])
        Rx = np.matrix([[1 ,  0,  0],
                        [0 , CR, -SR],
                        [0 , SR,  CR]])
        Ryc = np.matrix([[CPc, 0 ,  -SPc],
                        [0 ,  1,  0 ],
                        [SPc, 0,  CPc]])
        Rxc = np.matrix([[1 ,  0,  0],
                        [0 , CRc, -SRc],
                        [0 , SRc,  CRc]])
        R = Rz*Ry*Rx*Rxc.T*Ryc.T
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
    

class AttitudeCorrectionHardIron(AttitudeCorrection):
    
    def __init__(self, Hvector, declination, inclination):
        super().__init__()
        hard_iron = rdi_hardiron.HardIron(Hvector, declination, inclination)
        self.attitude_correction = hard_iron.attitude_correction

        
    
class Injector(object):
    def __init__(self):
        self.data={}

    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def set_data(self, section, name, data):
        '''
        Sets data from a given time series, so that these data can be inserted into the ensembles.

        Parameters 
        ----------
        section : string 
            string denoting the name of the section (example "variable_leader")
        name : string
             string name of the variable
        data : tuple   
            time and values
        '''
        self.data[(section, name)] = interp1d(*data)

    def gen(self, ensembles):
        for ens in ensembles:
            ens_tm = ens['variable_leader']['Timestamp']
            for (s,v), f in self.data.items():
                ens[s][v] = f(ens_tm) # need to do something when
                                      # we're interpolating beyond the
                                      # domain. For now an error will
                                      # occur...
            yield ens
        
class ReadAhead(object):
    def __init__(self, window_length, centered=False):
        self.window_length = window_length
        self.required_length = window_length
        if centered:
            self.required_length//=2
        
    def __call__(self,g):
        return self.gen(g)
        
    def gen(self, g):
        self.in_q = deque(maxlen = self.window_length)
        memo = []
        n = 0
        for v in g:
            memo.append(v)
            r = self.process(v)
            if r is None:
                n+=1
                continue
            s = memo.pop(0)
            yield s,r
        # complete backlog
        for i in range(n):
            r = self.process(None)
            s = memo.pop(0)
            yield s, r

    def process(self, v):
        if not v is None:
            self.in_q.append(v)
        else:
            try:
                self.in_q.popleft()
            except IndexError:
                return None
        if len(self.in_q)<self.required_length:
            return None
        return list(self.in_q)


class AdvanceAttitudeAngles(object):
    ''' Advance attitude data

    It has been observed that the attitude data, stored in the ensembles
    are usually about 8 seconds old. It seems that the other data are in 
    fact in sync with the glider data. This class implements a generator 
    to correct the pitch, heading and roll angles, reported in each ensemble.
    
    Parameters
    ----------
    dt : float
         (positive) time that pitch angles are delayed.
    window_length : int
         the number of pings that should be looked ahead

    Note
    ----
    The window_length should be large enough that it contains pings at least
    dt seconds later than the current one. This depends on the sampling interval.
    '''
    
    def __init__(self, dt, window_length):
        self.dt = dt
        self.read_ahead = ReadAhead(window_length)
        
    def __call__(self, g):
        for ens in self.gen_interpolate_angles(self.read_ahead(g)):
            yield ens

    def gen_interpolate_angles(self, s):
        ''' A generated that updates the times of the attitude angles

        Parameters
        ----------
        s : generator ReadAhead
        '''
        for ens, queue in s:
            if not queue is None:
                t = [q['variable_leader']['Timestamp'] for q in queue]
                p = [q['variable_leader']['Pitch'] for q in queue]
                ens['variable_leader']['Pitchp'] = np.interp(t[0]+self.dt, t, p)
                yield ens
            else:
                continue


    
class COG_Offset_Correction(object):

    def __init__(self, offsets = (0, 0.35, -0.11)):
        self.offsets = offsets
        
    def __call__(self, ensembles):
        return self.correct_angular_motion(ensembles)
    
            
    def correct_angular_motion(self, ensembles):
        rx, ry, rz = self.offsets
        read_ahead = ReadAhead(3)
        for ens, ens_context in read_ahead(ensembles):
            if ens['fixed_leader']['CoordXfrm']!='Ship':
                raise ValueError('To correct the angular motion, the coordinate system must be SHIP.')
            heading = [_ens['variable_leader']['Heading']*np.pi/180. for _ens in ens_context]
            pitch = [_ens['variable_leader']['Pitch']*np.pi/180. for _ens in ens_context]
            roll = [_ens['variable_leader']['Roll']*np.pi/180. for _ens in ens_context]
            dt = np.gradient([_ens['variable_leader']['Timestamp'] for _ens in ens_context])

            
            omega_x = (np.gradient(pitch)/dt)[1]
            omega_y = (np.gradient(roll)/dt)[1]
            omega_z = (np.gradient(heading)/dt)[1]

            ens['variable_leader']['omega_x'] = omega_x
            ens['variable_leader']['omega_y'] = omega_y
            ens['variable_leader']['omega_z'] = omega_z

            # using cross product:
            du = omega_y*rz - omega_z*ry
            dv = omega_z*rx - omega_x*rz
            dw = omega_x*ry - omega_y*rx
            ens['variable_leader']['Delta_U_sb'] = du
            ens['variable_leader']['Delta_U_f']  = dv
            ens['variable_leader']['Delta_U_u']  = dw
            # du, dv, dw are the velocity components of the sensor.
            # if still water and omega_z >0, ry >0, then du <0, that is
            # the sensor moves left. The recorded speed is water coming from the left, that is
            # u >0. So we need u (>0) + du (<0) to get 0.
            
            ens['velocity']['Velocity1']+=du 
            ens['velocity']['Velocity2']+=dv
            ens['velocity']['Velocity3']+=dw
            ens['bottom_track']['BTVel1']+=du
            ens['bottom_track']['BTVel2']+=dv
            ens['bottom_track']['BTVel3']+=dw
            yield ens




