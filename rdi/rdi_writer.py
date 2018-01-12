from collections import defaultdict
import datetime
import glob
import os
import sys

import numpy as np
from  netCDF4 import Dataset

import ndf
from rdi import __VERSION__
from rdi.rdi_reader import get_ensemble_time, unixtime_to_RTC

TransformationTranslations = dict(Earth = 'east north up error'.split(),
                                  Ship = 'starboard forward up error'.split(),
                                  Instrument = 'x y z error'.split(),
                                  Beam = 'beam1 beam2 beam3 beam4'.split())

def rad(x):
    return x*np.pi/180.


class Writer(object):
    YEAR = 2000
    def __init__(self):
        self.output_file = None
        self.custom_parameters = dict(scalar=[], vector=[])
        self.set_custom_parameter('sigma', '*', dtype='vector')
        
    def __call__(self, ensembles):
        self.write_ensembles(ensembles)
        
    def write_ensembles(self, ensembles):
        try:
            with open(self.output_file, 'w') as fd:
                self.__write_ensembles(fd, ensembles)
        except TypeError:
            self.__write_ensembles(self.output_file, ensembles)

    def set_custom_parameter(self, section, *name, dtype='scalar'):
        for _name in name:
            self.custom_parameters[dtype].append((section, _name))
            
    def __write_ensembles(self, fd, ensembles):
        config = None
        data1d = defaultdict(lambda : [])
        data2d = defaultdict(lambda : [])
        for ens in ensembles:
            if not config:
                config = ens['fixed_leader']
                self.write_configuration(config, fd)
                self.write_header(config, fd)
            self.read_variable_leader(data1d,ens)
            self.read_onedimdata(data1d, ens)
            self.read_twodimdata(data2d, ens)
            self.write_array(config, data1d, data2d, fd)
            data1d.clear()
            data2d.clear()
        

    def read_variable_leader(self, data, ens):
        vld = ens['variable_leader']
        try:
            tm = ens['variable_leader']['Timestamp']
        except KeyError:
            tm = get_ensemble_time(ens)
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
        # add any customized parameters.
        for s, p in self.custom_parameters['vector']:
            if p=="*":
                for k, v in ens[s].items():
                    try:
                        data[k].append(v)
                    except KeyError:
                        pass
            else:
                try:
                    data[p].append(ens[s][p])
                except KeyError:
                    pass

    def read_onedimdata(self, data, ens):
        try: # see if we have bottom track data, if not, ignore.
            bottom_track = ens['bottom_track']
        except KeyError:
            pass
        else:
            data['btv1'].append(bottom_track['BTVel1'])
            data['btv2'].append(bottom_track['BTVel2'])
            data['btv3'].append(bottom_track['BTVel3'])
            data['btv4'].append(bottom_track['BTVel4'])
            data['btpg1'].append(bottom_track['PG1'])
            data['btpg2'].append(bottom_track['PG2'])
            data['btpg3'].append(bottom_track['PG3'])
            data['btpg4'].append(bottom_track['PG4'])
        # add any customized parameters.
        for s, p in self.custom_parameters['scalar']:
            data[p].append(ens[s][p])

    def is_masked_array(self,v):
        ''' check whether v is a masked array'''
        g = (_v for _v in v)
        ma = False
        for _v in g:
            if isinstance(_v, np.ma.core.MaskedArray):
                ma = True
                break
        return ma
            
    def array2d_from_list(self,v):
        ''' return list v as an array or masked_array, depening on wheterh v is masked or not '''
        if self.is_masked_array(v):
            return np.ma.vstack(v)
        else:
            return np.vstack(v)

    def array1d_from_list(self,v):
        ''' return list v as an array or masked_array, depening on whether v has any nan's '''
        condition = np.isnan(v)
        if np.any(condition):
            return np.ma.masked_array(v, condition)
        else:
            return np.array(v)
        
        # subclass this class and implement these methods below.
        def write_configuration(self, config, fd):
            raise NotImplementedError("This method is not implemented. Subclass this class...")

        def write_header(self, config, fd):
            raise NotImplementedError("This method is not implemented. Subclass this class...")

        def write_array(self,config, data1d, data2d, fd):
            raise NotImplementedError("This method is not implemented. Subclass this class...")

# NetCDF format specifiers:
# just for reference...        

# 'f4' (32-bit floating point),
# 'f8' (64-bit floating point),
# 'i4' (32-bit signed integer),
# 'i2' (16-bit signed integer),
# 'i8' (64-bit signed integer),
# 'i1' (8-bit signed integer),
# 'u1' (8-bit unsigned integer),
# 'u2' (16-bit unsigned integer),
# 'u4' (32-bit unsigned integer),
# 'u8' (64-bit unsigned integer),
# 'S1' (single-character string). 



class NetCDFWriter(Writer):
    VARIABLES = dict(N_Cells = ('u1', 'scalar', '-'), 
                     N_PingsPerENS = ('u1', 'scalar', '-'),
                     Blank = ('f4', 'scalar', 'm'),
                     FirstBin = ('f4', 'scalar', 'm'),
                     #
                     Roll = ('f4', 'onedim', 'deg'), 
                     Pitch = ('f4', 'onedim', 'deg'), 
                     Heading = ('f4', 'onedim','deg'), 
                     Soundspeed = ('f4', 'onedim', 'm/s'), 
                     Salin = ('f4', 'onedim', 'SA'), 
                     Temp = ('f4', 'onedim', 'Celcius'), 
                     Press = ('f4', 'onedim', 'dbar?'), 
                     Ensnum = ('i8', 'onedim', '-'), 
                     Time = ('f8', 'onedim', 's'),
                     #
                     Velocity1 = ('f4', 'twodim', 'm/s'), 
                     Velocity2 = ('f4', 'twodim', 'm/s'), 
                     Velocity3 = ('f4', 'twodim', 'm/s'), 
                     Velocity4 = ('f4', 'twodim', 'm/s'), 
                     Echo1 = ('f4', 'twodim', 'dB'), 
                     Echo2 = ('f4', 'twodim', 'dB'), 
                     Echo3 = ('f4', 'twodim', 'dB'), 
                     Echo4 = ('f4', 'twodim', 'dB'), 
                     Echo_AVG = ('f4', 'twodim', 'dB'))
    
    SECTIONS = 'fixed_leader variable_leader velocity echo'.split()
                                    
                                    
    def __init__(self, output_file, ensemble_size_limit=None):
        ''' Constructor

        Parameters:
        -----------
        output_file: string representing the output filename
        ensemble_size_limit: integer | None limiting the number of ensembles to be written into 
                             a single netcdf file. None or 0 means no limit (one file will be written).
        '''
        
        super().__init__()
        self.ensemble_size_limit = ensemble_size_limit
        self.output_file = output_file
        
    def close(self):
        ''' Close current open file '''
        self.dataset.close()

    def write_ensembles(self, ensembles):
        '''Write ensembles to file or files

        Parameters:
        -----------

        ensembles: iteratable of ensembles (generator/list/etc)

        Returns:
        --------
        a tuple of the number of ensembles written and the number of files written.

        '''
        n = 0
        if self.ensemble_size_limit:
            self.open(n)
        else:
            self.open()
            
        k=0
        i=0
        for i, ens in enumerate(ensembles):
            if k==0:
                dimensions, variables = self.initialise(ens)
            self.add_ensemble(k, ens, variables)
            k+=1
            if self.ensemble_size_limit and k==self.ensemble_size_limit:
                self.close()
                n+=1
                k=0
                self.open(n)
        return i, n+1
    
    # "private" methods
    def create_dimensions(self, n_bins):
        time = self.dataset.createDimension('time', None)
        lat = self.dataset.createDimension('lat',1)
        lon = self.dataset.createDimension('lon',1)
        z = self.dataset.createDimension('z', n_bins)
        
        return time, z, lat, lon
    
    def create_variables(self, ens):
        variables =dict()
        variables['time'] = self.dataset.createVariable('time', 'f8', ('time',))
        variables['time'].units = 's'
        variables['time'].long_name = 'time'
        variables['time'].standard_name = 'time'

        
        variables['z'] = self.dataset.createVariable('z', 'f4', ('z',))
        variables['z'].units = 'm'
        z = np.arange(ens['fixed_leader']['N_Cells'])*ens['fixed_leader']['DepthCellSize']
        z += ens['fixed_leader']['FirstBin']
        variables['z'][...] = z

        for s, grp in ens.items():
            if s not in self.SECTIONS:
                continue
            for v, value in grp.items():
                try:
                    fmt, dim, units = self.VARIABLES[v]
                except KeyError:
                    pass
                else:
                    if dim == 'scalar':
                        variables[v] = self.dataset.createVariable(v, fmt)
                        variables[v][...] = value
                    elif dim == 'onedim':
                        variables[v] = self.dataset.createVariable(v, fmt, ('time',))
                    elif dim == 'twodim':
                        variables[v] = self.dataset.createVariable(v, fmt, ('time','z'))
                    else:
                        raise ValueError('Unknown Dimension specification.')
                    variables[v].units = units
        return variables

    def add_ensemble(self, k, ens, variables):
        for s, grp in ens.items():
            if s not in self.SECTIONS:
                continue
            for v, value in grp.items():
                try:
                    fmt, dim, units = self.VARIABLES[v]
                except KeyError:
                    pass
                else:
                    if dim == 'onedim':
                        variables[v][k] = value
                    elif dim == 'twodim':
                        variables[v][k,...] = value
        try:
            variables['time'][k] = ens['variable_leader']['Timestamp']
        except KeyError:
            variables['time'][k] = get_ensemble_time(ens)
    
    def initialise(self, ens):
        n_bins = ens['fixed_leader']['N_Cells']
        dimensions = self.create_dimensions(n_bins)
        variables = self.create_variables(ens)
        return dimensions, variables

    def open(self, n=None):
        if n==None:
            output_file=self.output_file
        else:
            base, ext = os.path.splitext(self.output_file)
            output_file = "{:s}{:04d}{:s}".format(base, n, ext)
        try:
            self.close()
        except:
            pass
        self.dataset = Dataset(output_file, 'w', format = "NETCDF4")
        

class AsciiWriter(Writer):
    DESCRIPTIONS = {"Earth":"Eastward current-Northward current-Upward current-Error velocity".split("-"),
                    "Beam" :"Beam 1-Beam 2-Beam 3-Beam 4".split("-"),
                    "Ship" :"Starboardward current-Forward current-Upward current- Error velocity".split("-"),
                    "Instrument": "Current in x direction-Current in y direction-Current in z direction-error velocity".split("-")}
                    
    def __init__(self, output_file = sys.stdout, adcp_offset=0):
        '''AsciiWriter
        
        Parameters:
        -----------
        output_file: a file pointer of filename (default sys.stdout)
        adcp_offset: the distance in m that the profile data should be offset by

        '''
        
        super().__init__()
        self.comment = "#"
        self.adcp_offset = adcp_offset
        self.output_file = output_file
        
  
    def write_configuration(self, config, fd=sys.stdout):
        kws = "Sys_Freq Xdcr_Facing N_Beams N_Cells N_PingsPerEns DepthCellSize Blank CoordXfrm WaterMode FirstBin SystemSerialNumber".split()
        kws_added = ['OriginalCoordXfrm'] #these get added during transformations etc.
        fd.write("{}Configuration:\n".format(self.comment))
        for kw in kws:
            fd.write("{}{} : {}\n".format(self.comment, kw, config[kw]))
        for kw in kws_added:
            try:
                fd.write("{}{} : {}\n".format(self.comment, kw, config[kw]))
            except KeyError:
                pass
        fd.write("{}\n".format(self.comment))
                     
    def write_header(self, config, fd=sys.stdout):
        ustr, vstr, wstr, verrstr = AsciiWriter.DESCRIPTIONS[config['CoordXfrm']]
        fd.write("{}{:19s}{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}\n".format(self.comment, "Time", "Elevation", ustr, vstr, wstr, verrstr))
        fd.write("{}{:19s}{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}\n".format(self.comment, "Date/Time", "m", "m/s", "m/s", "m/s", "m/s"))
        fd.write("{}\n".format(self.comment))
        
    def write_array(self, config, data1d, data2d, fd=sys.stdout):
        firstbin = config['FirstBin']
        binsize = config['DepthCellSize']
        factor = (int(config['Xdcr_Facing']=='Up')*2-1)
        for t, u, v, w, verr in zip(data1d['Time'], data2d['v1'], data2d['v2'], data2d['v3'], data2d['v4']):
            dt = datetime.datetime.utcfromtimestamp(t)
            tstr = dt.strftime("%Y-%m-%dT%H:%M:%S")
            for i, (_u, _v, _w, _verr) in enumerate(zip(u, v, w, verr)):
                r = firstbin + i*binsize
                z= factor*r + self.adcp_offset
                try:
                    fd.write("{:20s}{:< 20.3f}{:< 20.3f}{:< 20.3f}{:< 20.3f}{:< 20.3f}\n".format(tstr, z, _u, _v, _w, _v))
                except TypeError:
                    pass
                    

        
class NDFWriter(Writer):
    def __init__(self):
        super().__init__()
        self._global_parameters = dict()

    def write_ensembles(self, ensembles):
        ''' a non-lazy implementation. This reads all the data into memory because of
            how ndf files are written. NDF files cannot be written from generators.
        '''
        config = None
        data1d = defaultdict(lambda : [])
        data2d = defaultdict(lambda : [])
        
        for ens in ensembles:
            if not config:
                config = ens['fixed_leader']
            self.read_variable_leader(data1d,ens['variable_leader'])
            self.read_onedimdata(data1d, ens)
            self.read_twodimdata(data2d, ens)
        self.write_to_file(config, data1d, data2d)

    def write_to_file(self,config, data1d, data2d):
        data = self.create_ndf(config, data1d, data2d)
        data.save(self.output_file)


    def set_filename_from_pd0(self, filename_pd0,annotation=None):
        fn_base, fn_ext = os.path.splitext(filename_pd0)
        if annotation:
            self.output_file =  "{}-{}.ndf".format(fn_base,annotation)
        else:
            self.output_file =  "{}.ndf".format(fn_base)

        
    def add_global_parameter(self, key, value, unit):
        self._global_parameters[key] = value, unit
        
    
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
                i = int(k.replace("btv",""))-1
                s = TransformationTranslations[config['CoordXfrm']][i]
                ks = " ".join(["Bottom_track",s])
            elif k.startswith("btpg"):
                ks = k.replace("btpg","Bottom_track PercentGood")
            else:
                ks = k
            v = self.array1d_from_list(v)
            data.add_parameter(ks, units[k], (tm, v))
        for k, v in data2d.items():
            if k.startswith("v"):
                i = int(k.replace("v",""))-1
                s = TransformationTranslations[config['CoordXfrm']][i]
                ks = " ".join(["Velocity",s])
            elif k.startswith("e"):
                ks = k.replace("e","EchoIntensity")
            elif k.startswith("pg"):
                ks = k.replace("pg","PercentGood")
            else:
                ks = k
            v = self.array2d_from_list(v)
            data.add_parameter(ks, units[k], (tm,z,v.T))
        for k, v in config.items():
            if isinstance(v, str):
                data.add_metadata(k,v)
            else:
                data.add_global_parameter(k, v, units[k])
        # add any metadata present
        for k, v in self._global_parameters.items():
            data.add_global_parameter(k,*v)
        return data
