# ---- Last check on 07/12/2017 by Hades, the horrible --- #
''' 
First guide:


* import the rdi module with: - import rdi -
* create an PD0 object with: - pd0 = rdi.rdi_reader.PD0() -
* create the ensembles with: - ensembles = pd0.ensemble_generator('your_filenames_list') - 
* create an NW object with:  - nw = NumpyWriter()
* set custom variables with the method of same name if needed
* be happy with: - nw(ensembles) - .'''

from collections import defaultdict
import datetime
import numpy as np
from tqdm import tqdm

def rad(x):
    return x*np.pi/180.

class NumpyWriter(object):
    '''Creates an object whose attributes are ndarrays of the ADCP data
       read from the ensembles created by rdi_reader.'''
    YEAR = 2000

    _list = list([ 'Heading', 'Pitch', 'Roll', 'Salin', 'Temp'])
    
    _list2d = list(['v1','v2','v3','v4','e1','e2',
                    'e3','e4', 'pg1','pg2','pg3','pg4'])

    _list1d = list(['btv1','btv2','btv3','btv4',   
                    'btpg1','btpg2','btpg3','btpg4','time_adcp'])
       
    def __init__(self):
        self.custom_parameters = dict(scalar=[], vector=[])
        #self.set_custom_parameter('sigma', '*', dtype='vector')        
    def __call__(self, ensembles):
        config = None
        data1d = defaultdict(lambda : [])
        data2d = defaultdict(lambda : [])

        for string in self._list2d:
            self.__dict__[string]  = []
        for string in self._list1d:
            self.__dict__[string]  = []
        for string in self._list:
            self.__dict__[string]  = []
            
        for ens in tqdm(ensembles):
            if not config:
                config = ens['fixed_leader']
            self._read_variable_leader(data1d,ens['variable_leader'])
            self._read_onedimdata(data1d, ens)
            self._read_twodimdata(data2d, ens)
            self._write_dictionary(data1d, data2d)
            data1d.clear()
            data2d.clear()
        self._vectorize()
        
    def set_custom_parameter(self, key, *item, dtype='scalar'):
        ''' Set a custom parameter
        Set any parameter of interest contained in the dictionary of the ensembles pings. 
        The ensembles are the generator created by the rdi_reader module. 
        Custom parameters should be set before calling the main routine of the class.
        Keys and items are case sensitive and dtype as vector must be explicitly be specified.

        Parameters
        ----------
        key : string
           section name of the ensemble
        item : string
           name of variable
        dtype : string 
           data type (scalar|vector)
           
        Example
        -------
        >>> nw.NumpyWriter()          # create an instance
        >>> nw.set_custom_parameter('bottom_track','Corr1',dtype = 'vector')          
        >>> nw.set_custom_parameter('variable_leader','XdcrDepth',dtype = 'scalar')   
        >>> nw(ensembles)  # call the routine

        Values will be stored as methods in nw.bottom_track_Corr1 and nw.variable_leader_XdcrDepth'''
        for _name in item:
            self.custom_parameters[dtype].append((key, _name))
        if dtype == 'vector':
            self._list2d.append(key +'_'+ _name)
            self.__dict__[key + '_'+ _name] = []
        else:
            self._list.append(key +'_'+ _name)
            self.__dict__[key + '_'+ _name] = []
            
    def _read_variable_leader(self, data, vld):
        rtc = list(vld['RTC'])
        rtc[0]+=self.YEAR
        rtc[6]*=1000
        tm = datetime.datetime(*rtc, datetime.timezone.utc).timestamp()
        data['Ens'].append(vld['Ensnum'])
        data['time_adcp'].append(tm)
        data['Soundspeed'].append(vld['Soundspeed'])
        data['Depth'].append(vld['XdcrDepth'])
        data['Heading'].append(rad(vld['Heading']))
        data['Pitch'].append(rad(vld['Pitch']))
        data['Roll'].append(rad(vld['Roll']))
        data['Salinity'].append(vld['Salin'])
        data['Temperature'].append(vld['Temp'])
    
    def _read_twodimdata(self, data, ens):
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


        #add any customized parameters.
        for s, p in self.custom_parameters['vector']:

            string = s + '_' + p
            data[string].append(ens[s][p]) 

                        
    def _read_onedimdata(self, data, ens):
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
            
    def _is_masked_array(self,v):
        ''' check whether v is a masked array'''
        g = (_v for _v in v)
        ma = False
        for _v in g:
            if isinstance(_v, np.ma.core.MaskedArray):
                ma = True
                break
        return ma
            
    def _array2d_from_list(self,v):
        ''' return list v as an array or masked_array, depening on wheterh v is masked or not '''
        if self._is_masked_array(v):
            return np.ma.vstack(v)
        else:
            return np.vstack(v)

    def _array1d_from_list(self,v):
        ''' return list v as an array or masked_array, depening on whether v has any nan's '''
        condition = np.isnan(v)
        if np.any(condition):
            return np.ma.masked_array(v, condition)
        else:
            return np.array(v)
        
    def _write_dictionary(self, data1d, data2d):

        for string in self._list2d:
            self.__dict__[string].append(data2d[string])  
           
        for string in self._list1d:
            self.__dict__[string].append(data1d[string])

        for string in self._list:
            self.__dict__[string].append(data1d[string])
      
    def _vectorize(self):
        for string in self._list2d:
            self.__dict__[string]= (self._array2d_from_list(self.__dict__[string])).T 
           
        for string in self._list1d:
            self.__dict__[string]= np.squeeze(self._array1d_from_list(self.__dict__[string])  )

        for string in self._list:
            self.__dict__[string]= np.squeeze(self._array1d_from_list(self.__dict__[string])  )

# Hey Hades, do you really want to call this, when importing the module?            
#nw=NumpyWriter()

