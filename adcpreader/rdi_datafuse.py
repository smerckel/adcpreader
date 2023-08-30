from collections import OrderedDict
import glob
import os

import numpy as np

from adcpreader.coroutine import Coroutine, coroutine

    
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
    
