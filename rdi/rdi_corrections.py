import datetime

from scipy.interpolate import interp1d
import gsw

from rdi import __VERSION__

class SpeedOfSoundCorrection(object):
    Vhor = dict(velocity=['Velocity1', 'Velocity2'],
                bottom_track=['BTVel1', 'BTVel2', 'BTVel3'])
           
    def __init__(self, time_skew_threshold = None, RTC_year_base=2000):
        self.RTC_year_base = RTC_year_base
        self.time_skew_threshold = time_skew_threshold

    def get_ensemble_timestamp(self, ens):
        ''' returns timestamp in UTC in unix time (s)'''
        rtc = list(ens['variable_leader']['RTC'])
        rtc[0]+=self.RTC_year_base
        rtc[6]*=10000
        return datetime.datetime(*rtc, datetime.timezone.utc).timestamp()

    
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
            

