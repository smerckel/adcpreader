from collections import deque

import datetime
import numpy as np
from scipy.interpolate import interp1d
import gsw

from rdi import __VERSION__
from . import rdi_transforms
from rdi.rdi_reader import get_ensemble_time, unixtime_to_RTC


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

        t (s)    : unix time
        SA       : absolute salinity
        P (dbar) : pressure 
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

        t (s)    : unix time
        SA       : absolute salinity
        P (dbar) : pressure 
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

        Parameters:
        -----------
        ensembles:  list of ensembles or ensemble generator
        absolute_salinity: float
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

        Parameter:
        ----------
        
        aggregate_size: int, this many ensembles should be aggregated.
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
            
class ReadAhead(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def __call__(self,g):
        return self.gen(g)
        
    def gen(self, g):
        self.in_q = deque(maxlen = self.maxlen)
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
            yield s,r

    def process(self, v):
        if not v is None:
            self.in_q.append(v)
        else:
            try:
                self.in_q.popleft()
            except IndexError:
                return None
        if len(self.in_q)<=self.maxlen//2:
            return None
        return self.in_q
    
class PlatformAngularVelocityCorrection(object):

    def __init__(self, avg_window_size=5):
        self.avg_window_size = avg_window_size

    def get_attitude_rate(self, context_ens):
        n = len(context_ens)
        roll = np.empty(n, float)
        hdg = np.empty(n, float)
        pitch = np.empty(n, float)
        tm = np.empty(n, float)
        for i, ens in enumerate(context_ens):
            data = ens['variable_leader']
            tm[i] = get_ensemble_time(ens)
            hdg[i] = data['Heading']
            pitch[i] = data['Pitch']
            roll[i] = data['Roll']
        dt = np.gradient(tm)
        omega = np.array([np.gradient(pitch)/dt, np.gradient(roll)/dt, np.gradient(hdg)/dt])
        return omega.mean(axis=1)
            
    def correct_angular_motion(self, ensembles):
        read_ahead = ReadAhead(self.avg_window_size)
        n_cells = None
        FSU_ENU = rdi_transforms.RotationMatrix()
        ze = np.matrix([0,0,1.,0]).T
                       
        for ens, context_ens in read_ahead(ensembles):
            if not n_cells:
                n_cells = ens['fixed_leader']['N_Cells']
                bin_size = ens['fixed_leader']['DepthCellSize']
                first_bin = ens['fixed_leader']['FirstBin']
                z = np.arange(n_cells)*bin_size + first_bin
                if ens['fixed_leader']['CoordXfrm']!='Ship':
                    raise ValueError('To correct the angular motion, the coordinate system must be SHIP.')
            omega_x, omega_y, omega_z = self.get_attitude_rate(context_ens)
            R = FSU_ENU(omega_z, omega_x, omega_y)
            du, dv, dw, _ = [float(i) for i in  R * ze]
            
            
            ens['velocity']['Velocity1']+=z*du # positive because upward
                                             # pitch causes vy to be
                                             # more negative, so we
                                             # have to correct for
                                             # that.
            ens['velocity']['Velocity2']+=z*dv
            ens['velocity']['Velocity3']+=z*(1-dw)
            #print(du[2], dv[2], dw[2], z[2], omega_x, omega_y, omega_z)
            yield ens




