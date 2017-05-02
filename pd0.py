import os, time, struct
from collections import OrderedDict

import logging

import numpy as np

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
HRI["Tilts"]        = 2,1, ['','+Tilts']
HRI["3_Beam"]       = 1,1, ['','+3 Beam']
HRI["Bin_Mapping"]  = 0,1, ['','+Bin Mapping']
HRI["Bandwidth"]    = 0,2, ['BB', 'NB', 'na', 'na']

# data are stored as by, word (unsigned short), short and unsigned integer.
# the VARIABLE_DEFS dictionary lists the corresponding bit size and decode character.
VARIABLE_DEFS=dict(byte=(1,'B'), word=(2,'H'), short=(2,'h'), uint = (4,'I'))

# Conversion of counts to decibel. Should this be a hard coded constant?
ECHO_DB = 0.45


class Ensemble(object):
    '''
    class to hold and decode a binary data block containing a single ping.

    typical use:

    ens = Ensemble(bin_data)
    ens.decode()

    The constructor can take the data_offsets dictionary. If given, the offsets are not
    read from the binary data block, but assumed to be known.
    '''
    
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
            if block_id == 0x00:
                data['fixed_leader'] = self.decode_fixed_leader()
                n_cells = data['fixed_leader']['N_Cells']
                n_beams = data['fixed_leader']['N_Beams']
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
                data['bottom_track']=None
                logging.debug("Decoding bottom track: TODO")
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
        
    def get_string_from_byte(self, b, s):
        ''' method to convert a bit field value into human readbable information,
        as stored in the global constant dictionary HRI
        '''
        i, n, S = HRI[s]
        mask = 2**(n)-1
        idx = (b>>i) & mask
        return S[idx]
        

    def decode_variable_leader(self):
        '''
        decodes variable leader. 
        Returns dictionary with values.
        '''
        data =  OrderedDict()
        data['Ensnum'] = self.get_word()
        data['RTC'] = self.get_byte(n=7)
        data['Ensmsb'] = self.get_byte()
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
        data['Press'] = self.get_uint()
        data['PressVar'] = self.get_uint()
        self.get_byte() # skip spare byte
        data['RTCY2K'] = self.get_byte(n=8)
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
        v = np.array(self.get_byte(n = n_cells*n_beams), dtype=float) * ECHO_DB
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
        header['DepthCellSize'] = self.get_word()
        header['Blank'] = self.get_word()
        header['WaterMode'] = self.get_byte()
        header['CorrThresshold'] = self.get_byte()
        header['Code_Repts'] = self.get_byte()
        header['MinPG'] = self.get_byte()
        header['ErrVelThreshold'] = self.get_word()
        header['TimeBetweenPings'] = "{0:02d}:{1:02d}.{2:02d}".format(*self.get_byte(n=3))
        b = self.get_byte()
        header['RawCoordXrfm'] = b
        header['CoordXfrm'] = self.get_string_from_byte(b, 'CoordXfrm')
        header['CoordXfrm'] += self.get_string_from_byte(b, 'Tilts')
        header['CoordXfrm'] += self.get_string_from_byte(b, '3_Beam')
        header['CoordXfrm'] += self.get_string_from_byte(b, 'Bin_Mapping')
        for i in range(4):
            s = "Vel_field{:d}".format(i+1)
            header[s] = self.get_string_from_byte(b, s)
        header['EA'] = self.get_short()*1e-2
        header['EB'] = self.get_short()*1e-2
        header['Sensors'] = self.get_byte()
        header['Sensors_Avail'] = self.get_byte()
        header['FirstBin'] = self.get_word()*1e-2
        header['XmtLength'] = self.get_word()*1e-2
        header['WL_Start'] = self.get_byte()
        header['WL_End'] = self.get_byte()
        header['FalseTargetThreshold'] = self.get_byte()
        self.get_byte() # spare byte
        header['LagDistance'] = self.get_word()*1e-2
        header['CPUBoardSerial'] = " ".join(['{:02x}']*8).format(*self.get_byte(n=8))
        header['Bandwidth'] = self.get_string_from_byte(self.get_word(), 'Bandwidth')
        header['XmtPower'] = self.get_byte()
        return header
        
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
        Output: binary data in chunks, each containing the data on one ensemble.
        '''
        for fn in filenames:
            data = self.read(fn)
            yield from self.ensemble_data_generator(data)
            
    ### helper functions ###
    def get_word(self,data,idx):
        w, = struct.unpack('<H', data[idx:idx+2])
        return w

    def crc_check(self, data, idx, checksum_offset, checksum):
        crc = sum([i for i in data[idx:idx+checksum_offset]])
        crc %= 0x10000
        return crc == checksum
    
        
filename = "/home/lucas/gliderdata/tests/comet_ctd_noise/qc290849.pd0"
filename = "PF230519.PD0"

pd0 = PD0()
data = pd0.read(filename)
idx = [i for i in pd0.ensemble_data_generator(data)]
ens = Ensemble(idx[0])
data = ens.decode()

q=[Ensemble(i).decode() for i in pd0.ensemble_generator([filename])]



QQQQQQQQQQ
            
def find_possible_ens(filename):
    # Find possible starts and check if they represent valid ensembles
    possible_ens = []
    fi = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    # Main loop to find possible start bytes
    for fiby in range(filesize):
        if fiby == 0:
            hbyte1 = b""
            hbyte2 = ba.hexlify(fi.read(1))
        else:
            hbyte1 = hbyte2
            hbyte2 = ba.hexlify(fi.read(1))
        #  Header ID word for pd0 is 0x7f7f
        if hbyte2 + hbyte1 == b"7f7f":
            possible_ens.append(fi.tell() - 2) # stores possible start points
    fi.close()
    return(possible_ens)
        
def validify(filename, possible_ens):
    valid_ens = []
    fi = open(filename, 'rb')
    # Find valid Ensembles
    for element in possible_ens:
        chksum = 0
        fi.seek(element + 2, 0)
        # Determine offset to checksum by reading ensemble size in bytes
        chksum_offset = struct.unpack('<H', fi.read(2))[0]
        
        if chksum_offset >= 5000:
            print("dropped", element, chksum_offset)
            continue
        # Seek and read ensemble chksum out of file
        fi.seek(chksum_offset - 4, 1)
        RDI_chksum = struct.unpack('<H', fi.read(2))[0]
        fi.seek(element)
        for k in range(0, chksum_offset):
            chksum += struct.unpack('B', fi.read(1))[0]
        chksum = chksum % 65536
        if chksum == RDI_chksum:
            valid_ens.append(element)
        else:
            print("dropped still")
    fi.close()
    print("done")
    return(valid_ens)


#==============================================================================
# Define a structure similar to C++ or Matlab structs
#==============================================================================
class Ens(object):
    # Defines C++ struct like data array for storing the Ensemble Data
    def __init__(self, Ensnum, RTC, Ensmsb, BITResult, Soundspeed, XdcrDepth,
                 Heading, Pitch, Roll, Salin, Temp, MPT, Hdg_SD, Pitch_SD, 
                 Roll_SD, ADC, ErrorStatus, Press, PressVar, RTCY2K, Velocity1,
                 Velocity2, Velocity3, Velocity4, Corr1, Corr2, Corr3, Corr4,
                 Corr_AVG, Echo1, Echo2, Echo3, Echo4, Echo_AVG, PG1, PG2, PG3,
                 PG4):
        self.Ensnum = Ensnum
        self.RTC = RTC
        self.Ensmsb = Ensmsb
        self.BITResult = BITResult
        self.Soundspeed = Soundspeed
        self.XdcrDepth = XdcrDepth
        self.Heading = Heading
        self.Pitch = Pitch
        self.Roll = Roll
        self.Salin = Salin
        self.Temp = Temp
        self.MPT = MPT
        self.Hdg_SD = Hdg_SD
        self.Pitch_SD = Pitch_SD
        self.Roll_SD = Roll_SD
        self.ADC = ADC
        self.ErrorStatus = ErrorStatus
        self.Press = Press
        self.PressVar = PressVar
        self.RTCY2K = RTCY2K
        self.Velocity1 = Velocity1
        self.Velocity2 = Velocity2
        self.Velocity3 = Velocity3
        self.Velocity4 = Velocity4
        self.Corr1 = Corr1
        self.Corr2 = Corr2
        self.Corr3 = Corr3
        self.Corr4 = Corr4
        self.Corr_AVG = Corr_AVG
        self.Echo1 = Echo1
        self.Echo2 = Echo2
        self.Echo3 = Echo3
        self.Echo4 = Echo4
        self.Echo_AVG = Echo_AVG
        self.PG1 = PG1
        self.PG2 = PG2
        self.PG3 = PG3
        self.PG4 = PG4

class BT(object):
    # Defines C++ struct like data array for storing the Bottom Track Date
    def __init__(self, PPE, Delay, CorrMin, AmpMin, PGMin, Mode, ErrVelMax, 
                 Range1, Range2, Range3, Range4, BTVel1, BTVel2, BTVel3, 
                 BTVel4, Corr1, Corr2, Corr3, Corr4, Amp1, Amp2, Amp3, Amp4,
                 PG1, PG2, PG3, PG4, ReflMin, ReflNear, ReflFar, ReflVel1, 
                 ReflVel2, ReflVel3, ReflVel4, ReflCorr1, ReflCorr2, ReflCorr3,
                 ReflCorr4, ReflInt1, ReflInt2, ReflInt3, ReflInt4, ReflPG1,
                 ReflPG2, ReflPG3, ReflPG4, BTdepthMax, RSSI1, RSSI2, RSSI3,
                 RSSI4, Gain):
        self.PPE = PPE
        self.Delay = Delay
        self.CorrMin = CorrMin
        self.AmpMin = AmpMin
        self.PGMin = PGMin
        self.Mode = Mode
        self.ErrVelMax = ErrVelMax
        self.Range1 = Range1
        self.Range2 = Range2
        self.Range3 = Range3
        self.Range4 = Range4
        self.BTVel1 = BTVel1
        self.BTVel2 = BTVel2
        self.BTVel3 = BTVel3
        self.BTVel4 = BTVel4
        self.Corr1 = Corr1
        self.Corr2 = Corr2
        self.Corr3 = Corr3
        self.Corr4 = Corr4
        self.Amp1 = Amp1
        self.Amp2 = Amp2
        self.Amp3 = Amp3
        self.Amp4 = Amp4
        self.PG1 = PG1
        self.PG2 = PG2
        self.PG3 = PG3
        self.PG4 = PG4
        self.ReflMin = ReflMin
        self.ReflNear = ReflNear
        self.ReflFar = ReflFar
        self.ReflVel1 = ReflVel1
        self.ReflVel2 = ReflVel2
        self.ReflVel3 = ReflVel3
        self.ReflVel4 = ReflVel4
        self.ReflCorr1 = ReflCorr1
        self.ReflCorr2 = ReflCorr2
        self.ReflCorr3 = ReflCorr3
        self.ReflCorr4 = ReflCorr4
        self.ReflInt1 = ReflInt1
        self.ReflInt2 = ReflInt2
        self.ReflInt3 = ReflInt3
        self.ReflInt4 = ReflInt4
        self.ReflPG1 = ReflPG1
        self.ReflPG2 = ReflPG2
        self.ReflPG3 = ReflPG3
        self.ReflPG4 = ReflPG4
        self.BTdepthMax = BTdepthMax
        self.RSSI1 = RSSI1
        self.RSSI2 = RSSI2
        self.RSSI3 = RSSI3
        self.RSSI4 = RSSI4
        self.Gain = Gain
#==============================================================================
# Some internal functions
#==============================================================================
def tic():
    # Emulates the tic function of matlab
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
#=============================================================================#
def toc():
    #  Emulates the toc function of matlab
    if 'startTime_for_tictoc' in globals():
        t = str(time.time() - startTime_for_tictoc) 
    else:
        print("Toc: start time not set")
    return(t)
#=============================================================================#
def IDs():
    # Create dictionary with Block IDs
    global IDdict
    IDdict = {'fixed_leader' : b'0000',
              'variable_leader' : b'8000',
              'velocity' : b'0001',
              'correlation' : b'0002',
              'echo' : b'0003',
              'percent_good' : b'0004',
              'bottom_track' : b'0006',
              'nav' : b'2202'}
    return(IDdict)
#=============================================================================#
def find_possible_ens(filename):
    # Find possible starts and check if they represent valid ensembles
    possible_ens = []
    fi = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    # Main loop to find possible start bytes
    for fiby in range(filesize):
        if fiby == 0:
            hbyte1 = b""
            hbyte2 = ba.hexlify(fi.read(1))
        else:
            hbyte1 = hbyte2
            hbyte2 = ba.hexlify(fi.read(1))
        #  Header ID word for pd0 is 0x7f7f
        if hbyte2 + hbyte1 == b"7f7f":
            possible_ens.append(fi.tell() - 2) # stores possible start points
    fi.close()
    return(possible_ens)
#=============================================================================#
def validify(filename, possible_ens):
    valid_ens = []
    fi = open(filename, 'rb')
    # Find valid Ensembles
    for element in possible_ens:
        chksum = 0
        fi.seek(element + 2, 0)
        # Determine offset to checksum by reading ensemble size in bytes
        chksum_offset = struct.unpack('<H', fi.read(2))[0]
        
        if chksum_offset >= 5000:
            print("dropped", element, chksum_offset)
            continue
        # Seek and read ensemble chksum out of file
        fi.seek(chksum_offset - 4, 1)
        RDI_chksum = struct.unpack('<H', fi.read(2))[0]
        fi.seek(element)
        for k in range(0, chksum_offset):
            chksum += struct.unpack('B', fi.read(1))[0]
        chksum = chksum % 65536
        if chksum == RDI_chksum:
            valid_ens.append(element)
        else:
            print("dropped still")
    fi.close()
    print("done")
    return(valid_ens)
#=============================================================================#
def find_data_type_offset(filename, ensemble_start_byte):
    # Determine how many data types are in the next ensemble and save the start
    # bytes of each data type in vector. This is convenient to find the start
    # bytes for later processing
    fi = open(filename, 'rb')
    data_type_offsets = []
    # Byte 6 contains Number of data types in this ensemble
    fi.seek(ensemble_start_byte + 5)
    ndata_types = struct.unpack('B', fi.read(1))[0]
    # Following byte contain offset to data types
    for n in range(0, ndata_types):
        data_type_offsets.append(struct.unpack('<H', fi.read(2))[0])
    fi.close()
    return(ndata_types, data_type_offsets)
#=============================================================================#
def init_fixed_leader():
    # Initialize array for storage of Header Information
    Head = np.zeros(1, dtype = [('CPU_ver' ,'uint8'),
                                 ('CPU_rev', 'uint8'),
                                 ('Sys_Freq', 'a20'),
                                 ('Beam_Pattern', 'a20'),
                                 ('Sensor_Cfg', 'a20'),
                                 ('Xdcr_Head', 'a25'),
                                 ('Xdcr_Facing', 'a9'),
                                 ('Beam_Angle', 'a9'),
                                 ('Beam_Cfg', 'a25'),
                                 ('Real_Data', 'a9'),
                                 ('N_Beams', 'uint8'),
                                 ('N_Cells', 'uint8'),
                                 ('N_PingsPerEns', 'uint16'),
                                 ('DepthCellSize', 'uint16'),
                                 ('Blank', 'uint16'),
                                 ('WaterMode', 'uint8'),
                                 ('CorrThreshold', 'uint8'),
                                 ('Code_Repts', 'uint8'),
                                 ('MinPG', 'uint8'),
                                 ('ErrVelThreshold', 'uint16'),
                                 ('TimeBetweenPings', 'a8'),
                                 ('RawCoordXfrm', 'a8'),
                                 ('CoordXfrm', 'a40'),
                                 ('Vel_field1', 'a12'),
                                 ('Vel_field2', 'a12'),
                                 ('Vel_field3', 'a12'),
                                 ('Vel_field4', 'a12'),
                                 ('EA', 'float16'),
                                 ('EB', 'float16'),
                                 ('EZ', 'a8'),
                                 ('Sensors', 'a8'),
                                 ('Sensor_Avail', 'a8'),
                                 ('FirstBin', 'uint16'),
                                 ('XmtLength', 'uint16'),
                                 ('WL_Start', 'uint8'),
                                 ('WL_End', 'uint8'),
                                 ('FalseTargetThreshold', 'uint8'),
                                 ('CX', 'uint8'),
                                 ('LagDistance', 'float16'),
                                 ('CPUBoardSerial', 'a24'),
                                 ('Bandwidth', 'a2'),
                                 ('XmtPower', 'uint8')])
    return(Head)
#=============================================================================#
def init_Ens(valid_ens, Head):
    ens_cnt = len(valid_ens)
    DCN = Head['N_Cells'][0]
    Ensnum = np.zeros(ens_cnt, dtype= 'uint16')
    RTC = np.zeros((ens_cnt, 7), dtype = 'uint8')
    Ensmsb = np.zeros(ens_cnt, dtype = 'uint8')
    BITResult = np.zeros(ens_cnt, dtype = 'a17')
    Soundspeed = np.zeros(ens_cnt, dtype = 'uint16')
    XdcrDepth = np.zeros(ens_cnt, dtype = 'float16')
    Heading = np.zeros(ens_cnt, dtype = 'float16')
    Pitch = np.zeros(ens_cnt, dtype = 'float16')
    Roll = np.zeros(ens_cnt, dtype = 'float16')
    Salin = np.zeros(ens_cnt, dtype = 'uint16')
    Temp = np.zeros(ens_cnt, dtype = 'float16')
    MPT = np.zeros((ens_cnt, 3), dtype = 'uint8')
    Hdg_SD = np.zeros(ens_cnt, dtype = 'uint8')
    Pitch_SD = np.zeros(ens_cnt, dtype = 'uint8')
    Roll_SD = np.zeros(ens_cnt, dtype = 'uint8')
    ADC = np.zeros((ens_cnt, 8), dtype = 'uint8')
    ErrorStatus = np.zeros((ens_cnt, 4), dtype = 'a8')
    Press = np.zeros(ens_cnt, dtype = 'uint32')
    PressVar = np.zeros(ens_cnt, dtype = 'uint32')
    RTCY2K = np.zeros((ens_cnt, 8), dtype = 'uint8')
    Velocity1 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Velocity2 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Velocity3 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Velocity4 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Corr1 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    Corr2 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    Corr3 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    Corr4 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    Corr_AVG = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Echo1 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Echo2 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Echo3 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Echo4 = np.zeros((DCN, ens_cnt), dtype = 'float16')
    Echo_AVG = np.zeros((DCN, ens_cnt), dtype = 'float16')
    PG1 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    PG2 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    PG3 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    PG4 = np.zeros((DCN, ens_cnt), dtype = 'uint8')
    ens = Ens(Ensnum, RTC, Ensmsb, BITResult, Soundspeed, XdcrDepth, Heading,
              Pitch, Roll, Salin, Temp, MPT, Hdg_SD, Pitch_SD, Roll_SD, ADC, 
              ErrorStatus, Press, PressVar, RTCY2K, Velocity1, Velocity2,
              Velocity3, Velocity4, Corr1, Corr2, Corr3, Corr4, Corr_AVG, 
              Echo1, Echo2, Echo3, Echo4, Echo_AVG, PG1, PG2, PG3, PG4)
    return(ens)
#=============================================================================#
def init_BT(valid_ens, Head):
    ens_cnt = len(valid_ens)
#    DCN = Head['N_Cells'][0]
    PPE = np.zeros(ens_cnt, dtype = 'uint16')
    Delay = np.zeros(ens_cnt, dtype = 'uint16')
    CorrMin = np.zeros(ens_cnt, dtype = 'uint8')
    AmpMin  = np.zeros(ens_cnt, dtype = 'uint8')
    PGMin  = np.zeros(ens_cnt, dtype = 'uint8')
    Mode = np.zeros(ens_cnt, dtype = 'uint8')
    ErrVelMax = np.zeros(ens_cnt, dtype = 'float16')
    Range1 = np.zeros(ens_cnt, dtype = 'float16')
    Range2 = np.zeros(ens_cnt, dtype = 'float16')
    Range3 = np.zeros(ens_cnt, dtype = 'float16')
    Range4 = np.zeros(ens_cnt, dtype = 'float16')
    BTVel1 = np.zeros(ens_cnt, dtype = 'float16')
    BTVel2 = np.zeros(ens_cnt, dtype = 'float16')
    BTVel3 = np.zeros(ens_cnt, dtype = 'float16')
    BTVel4 = np.zeros(ens_cnt, dtype = 'float16')
    Corr1 = np.zeros(ens_cnt, dtype = 'uint8')
    Corr2 = np.zeros(ens_cnt, dtype = 'uint8')
    Corr3 = np.zeros(ens_cnt, dtype = 'uint8')
    Corr4 = np.zeros(ens_cnt, dtype = 'uint8')
    Amp1 = np.zeros(ens_cnt, dtype = 'uint8')
    Amp2 = np.zeros(ens_cnt, dtype = 'uint8')
    Amp3 = np.zeros(ens_cnt, dtype = 'uint8')
    Amp4 = np.zeros(ens_cnt, dtype = 'uint8')
    PG1 = np.zeros(ens_cnt, dtype = 'uint8')
    PG2 = np.zeros(ens_cnt, dtype = 'uint8')
    PG3 = np.zeros(ens_cnt, dtype = 'uint8')
    PG4 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflMin = np.zeros(ens_cnt, dtype = 'float16')
    ReflNear = np.zeros(ens_cnt, dtype = 'float16')
    ReflFar = np.zeros(ens_cnt, dtype = 'float16')
    ReflVel1 = np.zeros(ens_cnt, dtype = 'float16') 
    ReflVel2 = np.zeros(ens_cnt, dtype = 'float16')
    ReflVel3 = np.zeros(ens_cnt, dtype = 'float16')
    ReflVel4 = np.zeros(ens_cnt, dtype = 'float16')
    ReflCorr1 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflCorr2 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflCorr3 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflCorr4 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflInt1 = np.zeros(ens_cnt, dtype = 'float16')
    ReflInt2 = np.zeros(ens_cnt, dtype = 'float16')
    ReflInt3 = np.zeros(ens_cnt, dtype = 'float16')
    ReflInt4 = np.zeros(ens_cnt, dtype = 'float16')
    ReflPG1 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflPG2 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflPG3 = np.zeros(ens_cnt, dtype = 'uint8')
    ReflPG4 = np.zeros(ens_cnt, dtype = 'uint8')
    BTdepthMax = np.zeros(ens_cnt, dtype = 'float16')
    RSSI1 = np.zeros(ens_cnt, dtype = 'float16')
    RSSI2 = np.zeros(ens_cnt, dtype = 'float16')
    RSSI3 = np.zeros(ens_cnt, dtype = 'float16')
    RSSI4 = np.zeros(ens_cnt, dtype = 'float16')
    Gain = np.zeros(ens_cnt, dtype = 'uint8')
    BT_data = BT(PPE, Delay, CorrMin, AmpMin, PGMin, Mode, ErrVelMax, 
                 Range1, Range2, Range3, Range4, BTVel1, BTVel2, BTVel3, 
                 BTVel4, Corr1, Corr2, Corr3, Corr4, Amp1, Amp2, Amp3, Amp4,
                 PG1, PG2, PG3, PG4, ReflMin, ReflNear, ReflFar, ReflVel1, 
                 ReflVel2, ReflVel3, ReflVel4, ReflCorr1, ReflCorr2, ReflCorr3,
                 ReflCorr4, ReflInt1, ReflInt2, ReflInt3, ReflInt4, ReflPG1,
                 ReflPG2, ReflPG3, ReflPG4, BTdepthMax, RSSI1, RSSI2, RSSI3,
                 RSSI4, Gain) 
    return(BT_data)
#=============================================================================#
def decode_fixed_leader(filename, offset, Head):
    # Function to decode the fixed leader information and store them to 'Head'
    # Move to fixed leader and read first bytes to check truth
    fi = open(filename, 'rb')
    fi.seek(offset[0], 0)
    # Test for Ensemble Header ID 0x0000
    if ba.hexlify(fi.read(2)) == b"0000": # Fixed Leader
#        print 'Fixed Leader found' # Debug Information
        Head['CPU_ver'] = struct.unpack('<B', fi.read(1))[0]
        Head['CPU_rev'] = struct.unpack('<B', fi.read(1))[0]
        # read bitwise to decode system config
        byte1 = str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        byte2 = str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        # 3 LSB Bytes for System Frequency
        if byte1[5:] == b'000':
            Head['Sys_Freq'] = '75 kHz'
        elif byte1[5:] == b'001':
            Head['Sys_Freq'] = '150 kHz'
        elif byte1[5:] == b'010':
            Head['Sys_Freq'] = '300 kHz'
        elif byte1[5:] == b'011':
            Head['Sys_Freq'] = '600 kHz'
        elif byte1[5:] == b'100':
            Head['Sys_Freq'] = '1200 kHz'
        elif byte1[5:] == b'101':
            Head['Sys_Freq'] = '2400 kHz'
        else:
            Head['Sys_Freq'] = 'Not given'
        # 4th LSB for Beam Pattern:
        if byte1[4] == b'0':
            Head['Beam_Pattern'] = 'Concave'
        elif byte1[4] == b'1':
            Head['Beam_Pattern'] = 'Convex'
        else:
            Head['Beam_Pattern'] = 'Not given'
        # 5th to 6th LSB for Sensor Config
        if byte1[2:4] == b'00':
            Head['Sensor_Cfg'] = 'Sensor Cfg #1'
        elif byte1[2:4] == b'01':
            Head['Sensor_Cfg'] = 'Sensor Cfg #2'
        elif byte1[2:4] == b'10':
            Head['Sensor_Cfg'] = 'Sensor Cfg #3'
        else:
            Head['Sensor_Cfg'] = 'Not given'
        # 7th LSB for Beam Pattern:
        if byte1[1] == b'0':
            Head['Xdcr_Head'] = 'Xdcr Head not attached'
        elif byte1[1] == b'1':
            Head['Xdcr_Head'] = 'Xdcr Head attached'
        else:
            Head['Xdcr_Head'] = 'Not given'
        # 8th LSB for Beam Pattern:
        if byte1[0] == b'0':
            Head['Xdcr_Facing'] = 'Down'
        elif byte1[0] == b'1':
            Head['Xdcr_Facing'] = 'Up'
        else:
            Head['Xdcr_Facing'] = 'Not given'
        # 9th to 10th LSB for Sensor Config
        if byte2[6:] == b'00':
            Head['Beam_Angle'] = '15 Degree'
        elif byte2[6:] == b'01':
            Head['Beam_Angle'] = '20 Degree'
        elif byte2[6:] == b'10':
            Head['Beam_Angle'] = '30 Degree'
        else:
            Head['Beam_Angle'] = 'Not given'
        # 11th and 12th LSB is spare and does not contain information
        # Bits 13 to 16 for Beam Config
        if byte2[0:4] == b'0100':
            Head['Beam_Cfg'] = '4 Beam Janus'
        elif byte2[0:4] == b'0101':
            Head['Beam_Cfg'] = '5 Beam Janus w/ Demod'
        elif byte2[0:4] == b'1111':
            Head['Beam_Cfg'] = '5 Beam Janus w/ 2 Demod'
        else:
            Head['Beam_Cfg'] = 'Not given' 
        # Unpack next byte. Value of 0 means true Data, 1 for simulated
        byte1 = struct.unpack('B', fi.read(1))[0]
        if byte1 == 0:
            Head['Real_Data'] = 'True'
        elif byte1 == 1:
            Head['Real_Data'] = 'False'
        else:
            Head['Real_Data'] = 'Not given'
        # Skip next byte because it is spare
        fi.seek(1, 1)
        # Number of Beams
        Head['N_Beams'] = struct.unpack('B', fi.read(1))[0]
        # Number of Depth Cells
        Head['N_Cells'] = struct.unpack('B', fi.read(1))[0]
        # Pings per Ensemble
        Head['N_PingsPerEns'] = struct.unpack('<H', fi.read(2))[0]
        # Depth Cell Size
        Head['DepthCellSize'] = struct.unpack('<H', fi.read(2))[0]
        # Blank
        Head['Blank'] = struct.unpack('<H', fi.read(2))[0]
        # Water Mode
        Head['WaterMode'] = struct.unpack('B', fi.read(1))[0]
        # Correlation Threshold
        Head['CorrThreshold'] = struct.unpack('B', fi.read(1))[0]
        # Code Repititions
        Head['Code_Repts'] = struct.unpack('B', fi.read(1))[0]
        # Minimum % Good
        Head['MinPG'] = struct.unpack('B', fi.read(1))[0]
        # Error velocity threshold
        Head['ErrVelThreshold'] = struct.unpack('<H', fi.read(2))[0]
        # Pingtime
        timestring = ''
        for k in range(0,3):
            byte1 = struct.unpack('B', fi.read(1))[0]
            if k < 1:
                timestring = timestring + str(byte1).zfill(2) + ':'
            elif k >= 1 and k < 2:
                timestring = timestring + str(byte1).zfill(2) + '.'
            else:
                timestring = timestring + str(byte1).zfill(2)
        Head['TimeBetweenPings'] = timestring
        del timestring
        # Coordinate Xfrm
        byte1 = str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        Head['RawCoordXfrm'] = byte1
        # Decode RawCoordXform for better Readability for users
        Xfrm = ''
        if byte1[3:5] == b'00':
            Xfrm = 'Beam'
            Head['Vel_field1'] = 'To Beam 1'
            Head['Vel_field2'] = 'To Beam 2'
            Head['Vel_field3'] = 'To Beam 3'
            Head['Vel_field4'] = 'To Beam 4'
        elif byte1[3:5] == b'01':
            Xfrm = 'Instrument'
            Head['Vel_field1'] = 'Beam 1 - 2'
            Head['Vel_field2'] = 'Beam 4 - 3'
            Head['Vel_field3'] = 'To Xdcr'
            Head['Vel_field4'] = 'Error'
        elif byte1[3:5] == b'10':
            Xfrm = 'Ship'
            Head['Vel_field1'] = 'To Stbd'
            Head['Vel_field2'] = 'To Aft'
            Head['Vel_field3'] = 'Up'
            Head['Vel_field4'] = 'Error'
        elif byte1[3:5] == b'11':
            Xfrm = 'Earth'
            Head['Vel_field1'] = 'East'
            Head['Vel_field2'] = 'North'
            Head['Vel_field3'] = 'Up'
            Head['Vel_field4'] = 'Error'
        if byte1[5] == b'1':
            Xfrm = Xfrm + ' +Tilts'
        if byte1[6] == b'1':
            Xfrm = Xfrm + ' +3 Beam'
        if byte1[7] == b'1':
            Xfrm = Xfrm + ' +Bin Mapping'
        Head['CoordXfrm'] = Xfrm
        del Xfrm
        # Heading Alignment, i.e. Beam 3 Misalignment
        Head['EA'] = float(struct.unpack('<h', fi.read(2))[0])/100
        # Heading Bias, i.e. constant magnetic deviation
        Head['EB'] = float(struct.unpack('<h', fi.read(2))[0])/100
        # Sensor sources
        byte1 = str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        Head['Sensors'] = byte1
        byte1 = str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        Head['Sensor_Avail'] = byte1
        # First bin and transmit distance
        Head['FirstBin'] = float(struct.unpack('<H', fi.read(2))[0])/100
        Head['XmtLength'] = float(struct.unpack('<H', fi.read(2))[0])/100
        # Water Layer Bins
        Head['WL_Start'] = struct.unpack('B', fi.read(1))[0]
        Head['WL_End'] = struct.unpack('B', fi.read(1))[0]
        # False Target Threshold
        Head['FalseTargetThreshold'] = struct.unpack('B', fi.read(1))[0]
        # Jump spare byte
        fi.seek(1, 1)
        # Lag Distance
        Head['LagDistance'] = float(struct.unpack('<H', fi.read(2))[0])/100
        # CPU Board Serial
        Serial = ''
        for k in range(0, 8):
            byte1 = ba.hexlify(fi.read(1))
            Serial = Serial + ' ' +byte1.decode()
        Head['CPUBoardSerial'] = Serial
        # Bandwidth
        byte1 = struct.unpack('H', fi.read(2))[0]
        if byte1 == b'0':
            Head['Bandwidth'] = 'BB'
        elif byte1 == b'1':
            Head['Bandwidth'] = 'NB'
        else:
            Head['Bandwidth'] = 'na'
        # System Power
        Head['XmtPower'] = struct.unpack('B', fi.read(1))[0]
    fi.close()
    return(Head)
#=============================================================================#
def decode_main_data(filename, valid_ens, data_offset):
    # Main decoding work starting here
    fi = open(filename, 'rb')
    # Create dictionary with all the keys, temporary version. Can be refined 
    # quite a lot...
    block_offset = {}
    IDs()
    for element in data_offset:
        fi.seek(element, 0)
        twobytes = ba.hexlify(fi.read(2))
        if twobytes == IDdict['variable_leader']:
            block_offset['variable_leader'] = element
        elif twobytes == IDdict['velocity']:
            block_offset['velocity'] = element
        elif twobytes == IDdict['correlation']:
            block_offset['correlation'] = element
        elif twobytes == IDdict['echo']:
            block_offset['echo'] = element
        elif twobytes == IDdict['percent_good']:
            block_offset['percent_good'] = element
        else:
            continue
    del(twobytes)
    # Go to work now.
    eno = 0
    echo_db = 0.45
    # Main Loop for decoding (filename, valid_ens, data_offset)
    for ensemble in valid_ens:
        # Every ensemble has multiple data fields. This adresse them
        # Variable Leader, 0h8000
        fi.seek(ensemble + block_offset['variable_leader'] + 2, 0)
        ens.Ensnum[eno] = struct.unpack('<H', fi.read(2))[0]
        # RTC with 7 fields
        for i in range(0, 7):
            ens.RTC[eno, i] = struct.unpack('B', fi.read(1))[0]
        # The field Ensmsb indicates 1 if file has more than 65535 ensembles
        ens.Ensmsb[eno] = struct.unpack('B', fi.read(1))[0]
        ens.BITResult[eno] = (
                str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8) + 
                ' ' +
                str(bin(struct.unpack('B', fi.read(1))[0]))[2:].zfill(8))
        ens.Soundspeed[eno] = struct.unpack('<H', fi.read(2))[0]
        ens.XdcrDepth[eno] = struct.unpack('H', fi.read(2))[0] / 10 # dm -> m
        ens.Heading[eno] = float(struct.unpack('<H', fi.read(2))[0]) / 100
        ens.Pitch[eno] = float(struct.unpack('<h', fi.read(2))[0]) / 100
        ens.Roll[eno] = float(struct.unpack('<h', fi.read(2))[0]) / 100
        ens.Salin[eno] = struct.unpack('<H', fi.read(2))[0]
        ens.Temp[eno] = float(struct.unpack('<h', fi.read(2))[0]) / 100
        for i in range(0, 3):
            ens.MPT[eno, i] = struct.unpack('B', fi.read(1))[0]
        ens.Hdg_SD[eno] = struct.unpack('B', fi.read(1))[0]
        ens.Pitch_SD[eno] = struct.unpack('B', fi.read(1))[0]
        ens.Roll_SD[eno] = struct.unpack('B', fi.read(1))[0]
        for i in range(0, 8):
            ens.ADC[eno, i] = struct.unpack('B', fi.read(1))[0]
        for i in range(0, 4):
            ens.ErrorStatus[eno, i] = str(bin(
                    struct.unpack('B', fi.read(1))[0]))[2:].zfill(8)
        # Skip two bytes, reserved by RDI for future use
        fi.seek(2, 1)
        ens.Press[eno] = struct.unpack('<I', fi.read(4))[0]
        ens.PressVar[eno] = struct.unpack('<I', fi.read(4))[0]
        fi.seek(1, 1) # skip one spare byte
        for i in range(0, 8):
            ens.RTCY2K[eno, i] = struct.unpack('B', fi.read(1))[0]
        # data type 0h0001, Velocity
        fi.seek(ensemble + block_offset['velocity'], 0)
        fi.seek(2, 1) # skip ID bytes
        # Velocity Data, axis 0 is depth, axis 1 is ensembles. 
        # Meaning of velocity 1 - 4 is shown in Head['Vel_fieldx']
        for i in range(0, Head['N_Cells'][0]):
            ens.Velocity1[i, eno] = (
                    float(struct.unpack('<h', fi.read(2))[0]) / 1000)
            ens.Velocity2[i, eno] = (
                    float(struct.unpack('<h', fi.read(2))[0]) / 1000)
            ens.Velocity3[i, eno] = (
                    float(struct.unpack('<h', fi.read(2))[0]) / 1000)
            ens.Velocity4[i, eno] = (
                    float(struct.unpack('<h', fi.read(2))[0]) / 1000)
        # data type 0h0002, Correlation
        fi.seek(ensemble + block_offset['correlation'], 0) 
        fi.seek(2, 1) # skip ID bytes
        # Correlation Data, axis 0 is depth, axis 1 is ensembles.
        # Corr_AVG is a simple mean value of the four beams
        for i in range(0, Head['N_Cells'][0]):
            ens.Corr1[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.Corr2[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.Corr3[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.Corr4[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.Corr_AVG[i, eno] = np.mean((ens.Corr1[i, eno],
                        ens.Corr2[i, eno], ens.Corr3[i, eno],
                        ens.Corr4[i, eno]))
        # data type 0h0003, Echo Intensity
        fi.seek(ensemble + block_offset['echo'], 0) 
        fi.seek(2, 1) # skip ID bytes
        # Echo Data, axis 0 is depth, axis 1 is ensembles.
        # Echo Intensity is currently not calibrated. 0.45 db/count is just a 
        # guess. Average is a simple mean of all 4 values
        for i in range(0, Head['N_Cells'][0]):
            ens.Echo1[i, eno] = float(
                    struct.unpack('B', fi.read(1))[0]) * echo_db
            ens.Echo2[i, eno] = float(
                    struct.unpack('B', fi.read(1))[0]) * echo_db
            ens.Echo3[i, eno] = float(
                    struct.unpack('B', fi.read(1))[0]) * echo_db
            ens.Echo4[i, eno] = float(
                    struct.unpack('B', fi.read(1))[0]) * echo_db
            ens.Echo_AVG[i, eno] = np.mean((ens.Echo1[i, eno],
                        ens.Echo2[i, eno], ens.Echo3[i, eno],
                        ens.Echo4[i, eno]))
        # data type 0h0004, percent good    
        fi.seek(ensemble + block_offset['percent_good'], 0) 
        fi.seek(2, 1) # skip ID bytes
        # Percent Good Data, axis 0 is depth, axis 1 is ensembles.
        # For Meaning of the four fields see RDIs Output Data Format Manual
        for i in range(0, Head['N_Cells'][0]):
            ens.PG1[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.PG2[i, eno] = struct.unpack('B', fi.read(1))[0] 
            ens.PG3[i, eno] = struct.unpack('B', fi.read(1))[0]
            ens.PG4[i, eno] = struct.unpack('B', fi.read(1))[0]    
        # Next Ensemble iteration
        eno += 1
    # Replace bad velocity (-32768 mm/s) with nan
    ens.Velocity1[ens.Velocity1 == -32.768] = 'nan'    
    ens.Velocity2[ens.Velocity1 == -32.768] = 'nan'
    ens.Velocity3[ens.Velocity1 == -32.768] = 'nan'
    ens.Velocity4[ens.Velocity1 == -32.768] = 'nan'
    print('Coordinates are Velocity 1 to 4: ', Head['Vel_field1'][0],\
        Head['Vel_field2'][0], Head['Vel_field3'][0], Head['Vel_field4'][0])
    fi.close()
    del(eno)
    return(ens)
#=============================================================================#
def decode_BT(filename, valid_ens, data_offset):
    fi = open(filename, 'rb')
    eno = 0
    IDs()
    # finds the correct byte offset
    for element in data_offset:
        fi.seek(element, 0)
        twobytes = ba.hexlify(fi.read(2))
        if twobytes == IDdict['bottom_track']:
            BT_offset = element
        else:
            continue
        del(twobytes)
    # does the main work    
    for ensemble in valid_ens:
        # Seek start point of BT data and skip two ID bytes
        fi.seek(ensemble + BT_offset + 2, 0)
#        if eno == 1:
#            print BT_offset
        # Pings per Ensemble
        bt_data.PPE[eno] = struct.unpack('<H', fi.read(2))[0]
        # Delay befor reacquire in ensembles
        bt_data.Delay[eno] = struct.unpack('<H', fi.read(2))[0]
        # Correlatin Magnitude Min
        bt_data.CorrMin[eno] = struct.unpack('B', fi.read(1))[0]
        # Eval Amplitude Minimum
        bt_data.AmpMin[eno] = struct.unpack('B', fi.read(1))[0]
        # Minimum Percent Good
        bt_data.PGMin[eno] = struct.unpack('B', fi.read(1))[0]
        # Bottom Tracking Mode
        bt_data.Mode[eno] = struct.unpack('B', fi.read(1))[0]
        # Error Velocity Max in m/s
        bt_data.ErrVelMax[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        # skip reserved bytes
        fi.seek(4, 1)
        # BT ranges in m
        bt_data.Range1[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 100)
        bt_data.Range2[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 100)
        bt_data.Range3[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 100)
        bt_data.Range4[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 100)
        # BT velocities
        bt_data.BTVel1[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.BTVel2[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.BTVel3[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.BTVel4[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        # BT Beam Correlation Magnitudes
        bt_data.Corr1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Corr2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Corr3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Corr4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Beam Evaluation Amplitude
        bt_data.Amp1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Amp2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Amp3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.Amp4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Beam Percent Good
        bt_data.PG1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.PG2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.PG3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.PG4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Ref Layer Miniumum Layer Size
        bt_data.ReflMin[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 10)
        bt_data.ReflNear[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 10)
        bt_data.ReflFar[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 10)
        # BT Ref Layer Velocities
        bt_data.ReflVel1[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.ReflVel2[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.ReflVel3[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        bt_data.ReflVel4[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 1000)
        # BT Ref Layer Correlation Data
        bt_data.ReflCorr1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflCorr2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflCorr3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflCorr4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Ref Layer Intensity
        bt_data.ReflInt1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflInt2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflInt3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflInt4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Ref Layer Percent Good
        bt_data.ReflPG1[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflPG2[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflPG3[eno] = struct.unpack('B', fi.read(1))[0]
        bt_data.ReflPG4[eno] = struct.unpack('B', fi.read(1))[0]
        # BT Max depth two look for Bottom
        bt_data.BTdepthMax[eno] = (
                float(struct.unpack('<H', fi.read(2))[0]) / 10)
        # BT RSSI, rough guess, the 0.45 are approximate. This value differs
        # with every instrument and beam...
        bt_data.RSSI1[eno] = (
                float(struct.unpack('B', fi.read(1))[0]) * 0.45)
        bt_data.RSSI2[eno] = (
                float(struct.unpack('B', fi.read(1))[0]) * 0.45)
        bt_data.RSSI3[eno] = (
                float(struct.unpack('B', fi.read(1))[0]) * 0.45)
        bt_data.RSSI4[eno] = (
                float(struct.unpack('B', fi.read(1))[0]) * 0.45)
        # BT Gain Value
        bt_data.Gain[eno] = struct.unpack('B', fi.read(1))[0]
        eno += 1
    # Replace bad values -32768 mm/s with nan
    bt_data.BTVel1[bt_data.BTVel1 == -32.768] = 'nan'
    bt_data.BTVel2[bt_data.BTVel1 == -32.768] = 'nan'
    bt_data.BTVel3[bt_data.BTVel1 == -32.768] = 'nan'
    bt_data.BTVel4[bt_data.BTVel1 == -32.768] = 'nan'
    fi.close()
    del(eno)
    return(bt_data)
#==============================================================================
# Main Program
#==============================================================================

tic()
times = []
print("Started skript to decode raw pd0 files.")

# Only for Development purposes
filename = "C:/dev/adcp_bin/testfiles/PF260040.pd0"
filename = "/home/lucas/gliderdata/tests/comet_ctd_noise/qc290849.pd0"
filesize = os.path.getsize(filename)

# Search fpr possible start points
print("Looking for Ensembles in File ...")
possible_ens = find_possible_ens(filename)
times.append(float(toc())) # Time for finding possible starts


# Validation of ensembles via checksum
tic()
valid_ens = validify(filename, possible_ens)
ndata_types, data_offset = find_data_type_offset(filename, valid_ens[0])
times.append(float(toc())) # Time for validation
print("Found %d valid ensembles." %len(valid_ens))

# Decode Fixed Leader, one pass module
print("Decoding main Data ...")
tic()
Head = init_fixed_leader()
Head = decode_fixed_leader(filename, data_offset, Head)
times.append(float(toc())) # time decoding fixed leader

# Decode Main Data Sets, that is Leader (Ancillery), Velocity, Correlation, 
# Echo Intensity and Percent Good
tic()
ens = init_Ens(valid_ens, Head)
decode_main_data(filename, valid_ens, data_offset)
times.append(float(toc()))
print("Done")
tic()
print("Decoding Bottom Track Data ...")
bt_data = init_BT(valid_ens, Head)
decode_BT(filename, valid_ens, data_offset)
times.append(float(toc()))
print("Done")

# clear workspace from all auxiliary variables
del (filesize, possible_ens, startTime_for_tictoc, valid_ens,
     data_offset)

# Saves the main Variables to file
with open('c:/dev/adcp_bin/testfiles/test.pkl', 'wb') as output:
    pickle.dump((Head, ens, bt_data), output, -1)
