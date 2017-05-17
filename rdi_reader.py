import os, time, struct
from collections import OrderedDict

import logging

import numpy as np

# So das ist jetzt anders

# add filename=... to log to a file instead.
logging.basicConfig(level=logging.DEBUG)

ENSEMBLE_VARIABLES = """Ensnum RTC Ensmsb BITResult Soundspeed XdcrDepth
Heading Pitch Roll Salin Temp MPT Hdg_SD Pitch_SD 
Roll_SD ADC ErrorStatus Press PressVar RTCY2K Velocity1
Velocity2 Velocity3 Velocity4 Corr1 Corr2 Corr3 Corr4
Corr_AVG Echo1 Echo2 Echo3 Echo4 Echo_AVG PG1 PG2 PG3 PG4""".split()

HEX7F7F = b'\x7f\x7f' # ENSEMBLE START ID

SIZE_CHECKSUM = 2
POS_NUMBER_OF_DATA_TYPES = 0x05;

# HRI is a dictonary used to assign human-readable information
# corresponding to the various bits of information in the fixed
# leader.
# key: variable name to of the header dictonary
# value: 3 item tuple, consisting of
#         * start bit
#         * bit size of the field
#         * a '|' separated string with possible values.
# Example:
# if the bit size == 2, then there are 2**2 values, rangine from 0b00 0b01 0b10 0b11,
# or in decimal 0,1,2,3. The value corresponds to the index of the option string. So
# bit size  | number of options 
#    1               2
#    2               4
#    3               8
#    4              16
# Note there is NO checking if there are enough options.
HRI={}
HRI["Sys_Freq"]     = 0,3, '75 kHz|150 kHz|300 kHz|600 kHz|1200 kHz|2400 kHz|Not given'.split("|")
HRI["Beam_Pattern"] = 3,1, 'Concave|Convex'.split("|")
HRI["Sensor_Cfg"]   = 4,2, 'Sensor Cfg #1|Sensor Cfg #2|Sensor Cfg #3|Not given'.split("|")
HRI["Xdcr_Head"]    = 6,1, 'Xdcr Head not attached|Xdxr Head attached'.split("|")
HRI["Xdcr_Facing"]  = 7,1, 'Down|Up'.split("|")
HRI["Beam_Angle"]   = 0,2, '15 Degree|20 Degree|30 Degree|Not given'.split("|")
HRI["Beam_Cfg"]     = 4,4, 'x|x|x|x|4 Beam Janus|5 Beam Janus w/ Demod|x|x|x|x|x|x|x|x|x|5 Beam Janus w/ 2 Demod'.split("|")
HRI["Real_Data"]    = 0,1, ('True', 'False')
HRI["CoordXfrm"]    = 3,2, 'Beam|Instrument|Ship|Earth'.split("|")
HRI["Vel_field1"]   = 3,2, 'To Beam 1|Beam 1 - 2|To Stdbd|East'.split("|")
HRI["Vel_field2"]   = 3,2, 'To Beam 2|Beam 4 - 3|To Aft|North'.split("|")
HRI["Vel_field3"]   = 3,2, 'To Beam 3|To Xdcr|Up|Up'.split("|")
HRI["Vel_field4"]   = 3,2, 'To Beam 4|Error|Error|Error'.split("|")
HRI["Bandwidth"]    = 0,2, ['BB', 'NB', 'na', 'na']
HRI["FixedHeadingCoordinateFrame"] = 0,1, 'Instrument|Ship'.split("|")
HRI["Orientation"]  = 0,2, ["based on tilt", "fixed to up orientation", "fixed to down orientation"] 

# Descriptions where the bits in a byte denote possible options. So
# more than one option may apply.  To be queried using the
# get_or_strings_from_byte() method. Here the start bit is given as
# the first integer, the second integer denotes the number of options
# (bit size).

HRI["CoordXfrmOptions"] = 0,3, ['Bin Mapping', '3 Beam', 'Tilts']
HRI["Sensors"]      = 0,8, ["Uses EU from transducer temperature sensor",
                            "Uses ET from transducer temperature sensor",
                            "Uses ES (salinity) from conductivity sensor",
                            "Uses ER from transducer roll sensor",
                            "Uses EP from transducer pitch sensor",              
                            "Uses EH from transducer heading sensor",
                            "Uses ED from depth sensor",
                            "Calculates EC (speed of sound) from ED, ES, and ET"]


# data are stored as by, word (unsigned short), short and unsigned integer.
# the VARIABLE_DEFS dictionary lists the corresponding bit size and decode character.
VARIABLE_DEFS=dict(byte=(1,'B'), word=(2,'H'), short=(2,'h'), uint = (4,'I'))



class Ensemble(object):
    '''
    class to hold and decode a binary data block containing a single ping.

    typical use:

    ens = Ensemble(bin_data)
    ens.decode()

    The constructor can take the data_offsets dictionary. If given, the offsets are not
    read from the binary data block, but assumed to be known.
    '''
    # Conversion of counts to decibel.
    ECHO_DB = 0.45
    
    def __init__(self, bin_data, data_offsets=()):
        ''' constructor method

        bin_data: byte string containing a single ping
        data_offsets: dictionary with data offsets
        '''
        self.__data = bin_data
        self.__idx = None
        self.data_offsets = data_offsets or self.get_data_offsets()
        
    def decode(self):
        '''
        Method to decode a byte string.

        returns a dictionary with all the data decoded.
        '''
        data = {}
        # Can we assume that the data_offsets are always stored in increasing order?
        n_cells = None
        n_beams = None
        for offset in self.data_offsets:
            block_id = self.get_word(offset)
            self.__offset = offset
            if block_id == 0x00:
                data['fixed_leader'] = self.decode_fixed_leader()
                n_cells = data['fixed_leader']['N_Cells']
                n_beams = data['fixed_leader']['N_Beams']
            elif block_id == 0x0030:
                data['environmental_cmd_parameters'] = self.decode_environmental_command_parameters()
            elif block_id == 0x0080:
                data['variable_leader'] = self.decode_variable_leader()
            elif block_id == 0x0100:
                data['velocity'] = self.decode_velocity(n_cells, n_beams)
            elif block_id == 0x0200:
                data['correlation'] = self.decode_correlation(n_cells, n_beams)
            elif block_id == 0x0300:
                data['echo'] = self.decode_echo(n_cells, n_beams)
            elif block_id == 0x0400:
                data['percent_good'] = self.decode_percent_good(n_cells, n_beams)
            elif block_id == 0x0600:
                data['bottom_track'] = self.decode_bottom_track(n_beams)
            elif block_id == 0x2202:
                data['nav']=None
                logging.debug("Decoding nav: TODO")
            else:
                logging.info("Decoding block_id %08x not implemented."%(block_id))
        return data

    #### Helper functions ####
    def get_data_offsets(self):
        '''
        Returns a list of data offsets.
        '''
        n = self.get_byte(idx=POS_NUMBER_OF_DATA_TYPES)
        data_offsets = self.get_word(POS_NUMBER_OF_DATA_TYPES+1, n)
        return data_offsets

    def get_word(self,idx=None, n=1):
        ''' helper function to read a word (2 bytes). 
        if idx is given, it will be read from this position
        else the field following the last read is used.
        if n is given, then this number of words will be read.
        '''
        return self.get('word', idx, n)

    def get_byte(self,idx=None, n=1):
        ''' helper function to read a byte.
        if idx is given, it will be read from this position
        else the field following the last read is used.
        if n is given, then this number of words will be read.
        '''
        return self.get('byte', idx, n)

    def get_short(self, idx=None, n=1):
        ''' helper function to read a short (2 bytes). 
        if idx is given, it will be read from this position
        else the field following the last read is used.
        if n is given, then this number of words will be read.
        '''
        return self.get('short', idx, n)

    def get_uint(self, idx=None, n=1):
        ''' helper function to read a unsigned integer (4 bytes). 
        if idx is given, it will be read from this position
        else the field following the last read is used.
        if n is given, then this number of words will be read.
        '''
        return self.get('uint', idx, n)
    
    def get(self, dtype, idx=None, n=1):
        ''' helper function, not to be called directly. '''
        if idx==None:
            idx = self.__idx
        s, t = VARIABLE_DEFS[dtype]
        fmt = "<" + t*n
        self.__idx = idx + s*n
        w = struct.unpack(fmt, self.__data[idx:idx+n*s])
        if n==1:
            return w[0]
        else:
            return w

    @property    
    def current_position(self):
        ''' returns current position of read pointer '''
        return self.__idx - self.__offset # sets current byte position from data block.
    
    def get_string_from_byte(self, b, s):
        ''' method to convert a bit field value into human readbable information,
        as stored in the global constant dictionary HRI
        '''
        i, n, S = HRI[s]
        mask = 2**(n)-1
        idx = (b>>i) & mask
        return S[idx]
        
    def get_or_strings_from_byte(self, b, s):
        ''' method to convert a bit field value into human readbable information,
            as stored in the global constant dictionary HRI
            where each bit adds a valid option (as in x = V1 | V2 | V3 etc).
        '''
        i, n, S = HRI[s]
        m=[]
        for j in range(i, i+n):
            if (b>>j) & 1:
                m.append(S[j])
        return "|".join(m)
    
    def decode_fixed_leader(self):
        '''
        Decodes fixed leader good block.
        Returns a dictionary with values.
        '''
        header = OrderedDict()
        header['CPU_ver'] = self.get_byte()
        header['CPU_rev'] = self.get_byte()

        b = self.get_byte()
        for s in 'Sys_Freq Beam_Pattern Sensor_Cfg Xdcr_Head Xdcr_Facing'.split():
            header[s] = self.get_string_from_byte(b, s)
        b = self.get_byte()
        for s in 'Beam_Angle Beam_Cfg'.split():
            header[s] = self.get_string_from_byte(b, s)
        b = self.get_byte()
        header['Real_Data'] = self.get_string_from_byte(b, 'Real_Data')
        self.get_byte() # skip byte as it is spare
        header['N_Beams'] = self.get_byte() 
        header['N_Cells'] = self.get_byte() 
        header['N_PingsPerEns'] = self.get_word()
        header['DepthCellSize'] = self.get_word()*1e-2
        header['Blank'] = self.get_word()*1e-2
        header['WaterMode'] = self.get_byte()
        header['CorrThresshold'] = self.get_byte()
        header['Code_Repts'] = self.get_byte()
        header['MinPG'] = self.get_byte()
        header['ErrVelThreshold'] = self.get_word()*1e-3
        header['TimeBetweenPings'] = "{0:02d}:{1:02d}.{2:02d}".format(*self.get_byte(n=3))
        b = self.get_byte()
        header['RawCoordXrfm'] = b
        header['CoordXfrm'] = self.get_string_from_byte(b, 'CoordXfrm')
        header['CoordXfrmOptions'] = self.get_or_strings_from_byte(b, 'CoordXfrmOptions')
        for i in range(4):
            s = "Vel_field{:d}".format(i+1)
            header[s] = self.get_string_from_byte(b, s)
        header['EA'] = self.get_short()*1e-2
        header['EB'] = self.get_short()*1e-2
        header['Sensors'] = self.get_or_strings_from_byte(self.get_byte(), 'Sensors')
        header['Sensors_Avail'] = self.get_or_strings_from_byte(self.get_byte(), 'Sensors') # uses same list
                                                                                            #as Sensors!
        header['FirstBin'] = self.get_word()*1e-2
        header['XmtLength'] = self.get_word()*1e-2
        header['WL_Start'] = self.get_byte()
        header['WL_End'] = self.get_byte()
        header['FalseTargetThreshold'] = self.get_byte()
        self.get_byte() # spare byte
        header['LagDistance'] = self.get_word()*1e-2
        header['CPUBoardSerial'] = " ".join(['{:02x}']*8).format(*self.get_byte(n=8))
        header['Bandwidth'] = self.get_string_from_byte(self.get_word(), 'Bandwidth')
        header['XmtPower'] = self.get_byte() # DVL does not have
        self.get_byte() # spare
        header['SystemSerialNumber'] = self.get_uint()
        return header

    def decode_variable_leader(self):
        '''
        decodes variable leader. 
        Returns dictionary with values.
        '''
        data =  OrderedDict()
        data['Ensnum'] = self.get_word()
        data['RTC'] = self.get_byte(n=7)
        # reading Ensmsb and accounting its value directly in Ensnum
        Ensmsb = self.get_byte()
        data['Ensnum']+= Ensmsb * 0x10000
        data['BitResult'] = "{:08b} {:08b}".format(*self.get_byte(n=2))
        data['Soundspeed'] = self.get_word()
        data['XdcrDepth'] = self.get_word()*1e-2
        data['Heading'] = self.get_word()*1e-2
        data['Pitch'] = self.get_short()*1e-2
        data['Roll'] = self.get_short()*1e-2
        data['Salin'] = self.get_word()
        data['Temp'] = self.get_short()*1e-2
        data['MPT'] = self.get_byte(n=3)
        data['Hdg_SD'] = self.get_byte()
        data['Pitch_SD'] = self.get_byte()
        data['Roll_SD'] = self.get_byte()
        data['ADC'] = self.get_byte(n=8)
        data['ErrorStatus'] = "{:08b} {:08b} {:08b} {:08b}".format(*self.get_byte(n=4))
        self.get_byte(n=2) # skip two bytes
        data['Press'] = self.get_uint()*1e1    # ouput in Pa
        data['PressVar'] = self.get_uint()*1e1 # output in Pa
        self.get_byte() # skip spare byte
        data['RTCY2K'] = self.get_byte(n=8) # Glider DVL does not have this field, and exceeds the record.
        return data

    def decode_velocity(self, n_cells, n_beams):
        ''' 
        Decodes velocity block.
        Returns a dictionary with values.
        '''
        velocity = OrderedDict()
        v = np.array(self.get_short(n = n_cells*n_beams), dtype=float)*1e-3
        v = v.reshape(n_cells, n_beams).T
        for j in range(n_beams):
            k = 'Velocity%d'%(j+1)
            velocity[k] = v[j]
        return velocity
    
    def decode_correlation(self, n_cells, n_beams):
        '''
        Decodes correlations block.
        Returns a dictionary with values.
        '''
        correlation = OrderedDict()
        v = np.array(self.get_byte(n = n_cells*n_beams), dtype=float)
        v = v.reshape(n_cells, n_beams).T
        for j in range(n_beams):
            k = 'Corr%d'%(j+1)
            correlation[k] = v[j]
        correlation['Corr_AVG'] = v.mean(axis=0)
        return correlation
    
    def decode_echo(self, n_cells, n_beams):
        '''
        Decodes echo block.
        Returns a dictionary with values.
        '''
        echo = OrderedDict()
        v = np.array(self.get_byte(n = n_cells*n_beams), dtype=float) * self.ECHO_DB
        v = v.reshape(n_cells, n_beams).T
        for j in range(n_beams):
            k = 'Echo%d'%(j+1)
            echo[k] = v[j]
        echo['Echo_AVG'] = v.mean(axis=0)
        return echo

    def decode_percent_good(self, n_cells, n_beams):
        '''
        Decodes percent good block.
        Returns a dictionary with values.
        '''
        percent_good = OrderedDict()
        v = np.array(self.get_byte(n = n_cells*n_beams), dtype=float)
        v = v.reshape(n_cells, n_beams).T
        for j in range(n_beams):
            k = 'PG%d'%(j+1)
            percent_good[k] = v[j]
        return percent_good


    def decode_bottom_track(self, n_beams):
        bt = OrderedDict()
        bt['PPE'] = self.get_word()
        bt['Delay'] = self.get_word()
        bt['CorrMin'] = self.get_byte()
        bt['AmpMin'] = self.get_byte()
        bt['PGMin'] = self.get_byte()
        bt['Mode'] = self.get_byte()
        bt['ErrVelMax'] = self.get_word()*1e-3
        self.get_byte(n=4) # skip reserved bytes
        for k in range(n_beams):
            key = "Range{}".format(k+1)
            bt[key] = self.get_word()*1e-2
        for k in range(n_beams):
            key = "BTVel{}".format(k+1)
            bt[key] = self.get_short()*1e-3
        for k in range(n_beams):
            key = "Corr{}".format(k+1)
            bt[key] = self.get_byte()
        for k in range(n_beams):
            key = "Amp{}".format(k+1)
            bt[key] = self.get_byte()
        for k in range(n_beams):
            key = "PG{}".format(k+1)
            bt[key] = self.get_byte()
        bt['ReflMin'] = self.get_word()*1e-1
        bt['ReflNear'] = self.get_word()*1e-1
        bt['ReflFar'] = self.get_word()*1e-1
        for k in range(n_beams):
            key = "ReflVel{}".format(k+1)
            bt[key] = self.get_short()*1e-3
        for k in range(n_beams):
            key = "ReflCorr{}".format(k+1)
            bt[key] = self.get_byte()
        for k in range(n_beams):
            key = "ReflInt{}".format(k+1)
            bt[key] = self.get_byte()
        for k in range(n_beams):
            key = "ReflPG{}".format(k+1)
            bt[key] = self.get_byte()
        bt['BTdepthMax'] = self.get_word()*1e-1
        for k in range(n_beams):
            key = "RSSI{}".format(k+1)
            bt[key] = self.get_byte() * 0.45
        bt['Gain'] = self.get_byte()
        return bt

    # I thought it this data block would contain some useful information, but it turns out that the DVL
    # (at least) does not output it. This method has not been tested and therefore will raise an error.
    def decode_environmental_command_parameters(self):
        data = OrderedDict()
        data["AttitudeOutputCoordinate"] = self.get_byte(n=8)
        self.get_byte()
        data["FixedHeadingScaling"] = self.get_short()*1e-2
        data["FixedHeadingCoordinateFrame"] = self.get_string_from_byte(self.get_byte(),
                                                                        'FixedHeadingCoordinateFrame')
        data["RollMisalignment"] = self.get_short()*1e-2
        data["PitchMisalignment"] = self.get_short()*1e-2
        data["AttitudeCoordinateFrame"] = self.get_byte(n=4) #TODO
        data["Orientation"] = self.get_string_from_byte(self.get_byte(), 'Orientation')
        data["HeadingOffset"] = self.get_short()*1e-2
        data["SensorSource"] = self.get_byte(n=8) #TODO
        data["TransducerDepth"] = self.get_uint()*1e-1 # set in dm
        data['Salinity'] = self.get_byte()
        data['WaterTemp'] = self.get_short()*1e-2
        data['SpeedOfSound'] = self.get_word()
        data['Transform'] = self.get_byte() # TODO
        data['3BeamSolution'] = self.get_byte()
        data['BinMap'] = self.get_byte()
        data['MSB_EX_transformation'] = self.get_byte()
        raise RuntimeError('decode_environemental_command_parameters is NOT tested yet because of lack of data')
        return data
        

class PD0(object):
    ''' Class to process one or multiple PD0 files.
    '''
    
    def read(self, filename):
        ''' Read the contents of given filename and return binary data.
        '''
        with open(filename, 'rb') as fd:
            data = fd.read()
        return data
    
    def ensemble_data_generator(self, data):
        ''' Generator returning binary data per ensemble.

        Takes a binary data block as input

        returns binary data in chunks, each containing the data of one ensemble
        '''
        idx_next = 0
        while True:
            idx = data.find(HEX7F7F, idx_next)
            if idx==-1:
                break
            checksum_offset = self.get_word(data, idx+2)
            checksum = self.get_word(data, idx + checksum_offset)
            if not self.crc_check(data, idx, checksum_offset, checksum):
                logging.debug("CRC mismatch at 0x%x"%(idx))
                continue
            idx_next = idx + checksum_offset + SIZE_CHECKSUM
            yield data[idx:idx_next]

    def ensemble_generator(self, filenames):
        ''' Generator returnining binary data per ensemble for list of filenames
        
        Input: list of file names:
        Output: decoded ensemble.
        '''
        for fn in filenames:
            data = self.read(fn)
            for chunk in self.ensemble_data_generator(data):
                yield Ensemble(chunk).decode()
            
    ### helper functions ###
    def get_word(self,data,idx):
        w, = struct.unpack('<H', data[idx:idx+2])
        return w

    def crc_check(self, data, idx, checksum_offset, checksum):
        crc = sum([i for i in data[idx:idx+checksum_offset]])
        crc %= 0x10000
        return crc == checksum

if __name__ == "__main__":    
    filename = "/home/lucas/gliderdata/tests/comet_ctd_noise/qc290849.pd0"
    filename = "PF230519.PD0"
    
    pd0 = PD0()
    data = pd0.read(filename)
    idx = [i for i in pd0.ensemble_data_generator(data)]
    ens = Ensemble(idx[0])
    data = ens.decode()
    
    q=[i for i in pd0.ensemble_generator([filename])]
