from collections import OrderedDict
import glob
import os

import numpy as np

import ndf

import rdi
from rdi.coroutine import Coroutine, coroutine


class NDFReader(Coroutine):
    def __init__(self, filename):
        super().__init__()
        self.coro_fun = self.coro_read_from_ndf(filename)
        
    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        if type is None:
            self.close_coroutine()

    @coroutine
    def coro_read_from_ndf(self, filename):
        ''' 
        Coroutine reading from a single ndf file. Upon the receptioin of
        an ensemble the pointer is advanced until the appropriate time 
        that of the ensemble, and the u, v, w and z data from the glider flight model
        are send out as a new ordered dictionary.

        Parameters
        ----------
        filename : string
            name of ndf file containing gliderflight data

        Returns
        -------
        coroutine

        '''
        data = ndf.NDF(filename, open_mode='open')
        t, u, v, w, z = data.get_sync("u", "v w z".split())
        data.close()
        i = 0
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                t_adcp = ens['variable_leader']['Timestamp']
                while t[i]<=t_adcp:
                    i+=1
                f = (t_adcp-t[i-1])/(t[i]-t[i-1])
                ui = f * u[i] + (1-f)*u[i-1]
                vi = f * v[i] + (1-f)*v[i-1]
                wi = f * w[i] + (1-f)*w[i-1]
                zi = f * z[i] + (1-f)*z[i-1]
                gf = OrderedDict()
                gf['velocity_east'] = ui
                gf['velocity_north'] = vi
                gf['velocity_up'] = wi
                gf['depth'] = zi
                self.send(gf)
        self.close_coroutine()

    
class DataFuser(Coroutine):
    ''' Data fuser class

    Parameters
    ----------

    Channels : positional parameters containing the section name into which the dictionaries pertaining to
    each input channel should be written.


    It makes only sense to use this class if two or more components of the data pipeline structure push data
    into this object. The data from the first source is assumed to be the ensemble dictionary, which is to 
    be fused with data from other sources. 

    .. code-block:: text
    

         ____________                        
        |            |                                                    
        | pd0 reader |                       
        |            |                       
         ------------                        
             | |                             
             | +---------------------+       
             |                       |       
             +---------+        _____V______ 
                       |       |            |
                       |       | NDF reader |
                       |       |            |
          ensemble dict|        ------------ 
                       |             |       
                       |             |       
                       |  +----------+  glider flight dictionary      
                  _____V__V___               
                 |            |              
                 | data fuser |              
                 |            |              
                  ------------               
                       |                     
                       |                     
                       V                       


        ensemble dict with added glider flight dict

                          
    In this schematic, ensembles are pushed into the system by pd0 reader, and fed into datafuser and NDF reader.
    NDF reader uses its time stamp to find the correct data points from the ndf file, and sends this information
    as a dictionary to data fuser.

    Data fuser receives from two streams, first the original ensemble, and then added data. 
    Data fuser then outputs the ensemble dictionary with the other data dictionary inserted.

    If the name of the channel exists, the data are added to the already existing section dictionary, 
    otherwise a new section is created. For example, the glider flight data can be inserted as a new
    section "glider_flight", or the data could be added to an existing section, such as "variable_leader".

    '''
    def __init__(self, *channels):
        super().__init__()
        self.coro_fun = self.coro_read_data(*channels)
        
    @coroutine
    def coro_read_data(self, *channels):
        ''' coroutine that fuses that data into the main ensemble

        See also the notes of the class itself.
        
        Parameters
        ----------
        channels : positional parameters (strings)
                   name of sections

        Returns
        -------
        coroutine


        '''
        n = len(channels)
        data=[]
        while True:
            try:
                for i in range(1+n):
                    if i:
                        data.append( (yield) )
                    else:
                        ens = (yield)
            except GeneratorExit:
                break
            else:
                for _data, _channel in zip(data, channels):
                    self.merge(ens, _data, _channel)
                self.send(ens)
                data.clear()
        self.close_coroutine()
        
    def merge(self, ens, data, channel):
        ''' Merges data per channel into ens

        Parameters
        ----------
        ens : dictionary
              ensemble dictionary
        data : a dictionary from a channel
        channel : string
                  name of section where the data should be inserted into

        If ensemble already has the channel name as key, the dictionary is added to the
        exisiting section,  otherwise a new section is created.

        '''
        if channel in ens.keys():
            for k,v in data.items():
                ens[channel][k]=v
        else:
            ens[channel] = data
    
