from collections import defaultdict
import datetime
import glob

import numpy as np

import ndf

from . import rdi_reader
from . import rdi_corrections


def rad(x):
    return x*np.pi/180.

class Pd0NDF(object):
    YEAR = 2000
    def read_files(self, pattern):
        fns = glob.glob(pattern)
        fns.sort()
        return fns

    def read_config(self, fns):
        pd0 = rdi_reader.PD0()
        for ens in pd0.ensemble_generator(fns):
            break
        config = ens['fixed_leader']
        return config

    def read_variable_leader(self, data, vld):
        rtc = list(vld['RTC'])
        rtc[0]+=Pd0NDF.YEAR
        rtc[6]*=1000
        tm = datetime.datetime(*rtc, datetime.timezone.utc).timestamp()
        data['Ens'].append(vld['Ensnum'])
        data['Time'].append(tm)
        data['Soundspeed'].append(vld['Soundspeed'])
        data['Depth'].append(vld['XdcrDepth'])
        data['Heading'].append(rad(vld['Heading']))
        data['Pitch'].append(rad(vld['Pitch']))
        data['Roll'].append(rad(vld['Roll']))
        data['Salinity'].append(vld['Salin'])
        data['Temperature'].append(vld['Temp'])
    
    def read_twodimdata(self, data, ens):
        data['v1'].append(ens['velocity']['Velocity1'])
        data['v2'].append(ens['velocity']['Velocity2'])
        data['v3'].append(ens['velocity']['Velocity3'])
        data['v4'].append(ens['velocity']['Velocity4'])
        data['e1'].append(ens['echo']['Echo1'])
        data['e2'].append(ens['echo']['Echo2'])
        data['e3'].append(ens['echo']['Echo3'])
        data['e4'].append(ens['echo']['Echo4'])
        data['pg1'].append(ens['percent_good']['PG1'])
        data['pg2'].append(ens['percent_good']['PG2'])
        data['pg3'].append(ens['percent_good']['PG3'])
        data['pg4'].append(ens['percent_good']['PG4'])

    def read_onedimdata(self, data, ens):
        data['btv1'].append(ens['bottom_track']['BTVel1'])
        data['btv2'].append(ens['bottom_track']['BTVel2'])
        data['btv3'].append(ens['bottom_track']['BTVel3'])
        data['btv4'].append(ens['bottom_track']['BTVel4'])
        data['btpg1'].append(ens['bottom_track']['PG1'])
        data['btpg2'].append(ens['bottom_track']['PG2'])
        data['btpg3'].append(ens['bottom_track']['PG3'])
        data['btpg4'].append(ens['bottom_track']['PG4'])
        
        
                          
    def read_data(self, fns, ctd_data = None):
        data2d = defaultdict(lambda : [])
        data1d = defaultdict(lambda : [])
        pd0 = rdi_reader.PD0()
        config = self.read_config(fns)
        ensembles = pd0.ensemble_generator(fns)
        if ctd_data:
            current_corrector = rdi_corrections.SpeedOfSoundCorrection()
            ensemble_gen = current_corrector.horizontal_current_from_salinity_pressure(ensembles, *ctd_data)
        else:
            ensemble_gen = ensembles

        for i, ens in enumerate(ensemble_gen):
            self.read_variable_leader(data1d,ens['variable_leader'])
            self.read_onedimdata(data1d, ens)
            self.read_twodimdata(data2d, ens)
        return config, data1d, data2d
            
        
    def create_ndf(self, config, data1d, data2d):
        units=defaultdict(lambda : '-', Soundspeed='m/s', Temperature='degree', Depth='m',
                          v1='m/s',v2='m/s',v3='m/s',v4='m/s',
                          btv1='m/s',btv2='m/s',btv3='m/s',btv4='m/s',
                          e1='dB',e2='dB',e3='dB',e4='dB',
                          Beam_Angle='deg', DepthCellSize='m',Blank='m',ErrVelThreshold='m/s',
                          FirstBin='m',XmtLegnth='m', LagDistance='m')
        data = ndf.NDF()
        tm = np.array(data1d['Time'])
        n_cells = config['N_Cells']
        bin_size = config['DepthCellSize']
        z = np.arange(n_cells)*bin_size + config['FirstBin']
        for k, v in data1d.items():
            if k == 'Time':
                continue
            if k.startswith("btv"):
                ks = " ".join(["Bottom_track",config[k.replace("btv","Vel_field")]])
            elif k.startswith("btpg"):
                ks = k.replace("btpg","Bottom_track PercentGood")
            else:
                ks = k
            data.add_parameter(ks, units[k], (tm, np.array(v)))
        for k, v in data2d.items():
            if k.startswith("v"):
                ks=" ".join(["Velocity",config[k.replace("v","Vel_field")]])
            elif k.startswith("e"):
                ks = k.replace("e","EchoIntensity")
            elif k.startswith("pg"):
                ks = k.replace("pg","PercentGood")
            else:
                ks = k
            data.add_parameter(ks, units[k], (tm,z,np.array(v).T))
        for k, v in config.items():
            if isinstance(v, str):
                data.add_metadata(k,v)
            else:
                data.add_global_parameter(k, v, units[k])
        return data

