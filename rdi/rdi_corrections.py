from collections import deque

import datetime
import numpy as np
from scipy.interpolate import interp1d
import gsw

from rdi import __VERSION__
from . import rdi_transforms

def get_ensemble_timestamp(ens, century=2000):
    ''' returns timestamp in UTC in unix time (s)'''
    rtc = list(ens['variable_leader']['RTC'])
    rtc[0]+=century
    rtc[6]*=10000
    tm = datetime.datetime(*rtc, datetime.timezone.utc).timestamp()
    if tm<1e9:
        Q
    return tm



class SpeedOfSoundCorrection(object):
    Vhor = dict(velocity=['Velocity1', 'Velocity2'],
                bottom_track=['BTVel1', 'BTVel2', 'BTVel3'])
           
    def __init__(self, RTC_year_base=2000):
        self.RTC_year_base = RTC_year_base

    def get_ensemble_timestamp(self, ens):
        ''' returns timestamp in UTC in unix time (s)'''
        return get_ensemble_timestamp(ens,self.RTC_year_base)
    
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
            tm[i] = get_ensemble_timestamp(ens)
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




