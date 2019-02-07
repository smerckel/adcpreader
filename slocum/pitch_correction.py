import configobj
import glob
import os

import numpy as np
from matplotlib import pyplot, dates

from scipy.optimize import fmin

import dbdreader
from rdi import rdi_reader, rdi_transforms, rdi_corrections, rdi_writer, rdi_qc

    
def get_filenames(datadir, glider, n_files = None, start = 0, pd0_extension='pd0'):
    pattern = os.path.join(datadir, "adcp", "%s*.%s"%(glider,pd0_extension))
    filenames = glob.glob(pattern)
    filenames.sort()
    n_files = n_files or len(filenames)
    n_files = min(n_files, len(filenames)-start)
    dvl_filenames = filenames[start:start+n_files]
    #
    gld_filenames = [i.replace("adcp", "hd").replace(pd0_extension, "dbd") for i in dvl_filenames]
    if not len(dvl_filenames) or not len(gld_filenames):
        raise ValueError('Found %d dvl files and %d glider files'%(len(dvl_filenames), len(gld_filenames)))
    return dvl_filenames, gld_filenames



class DVL_GliderPitchBase(object):
        
    MOUNT_PITCH = 11*np.pi/180

    def __init__(self, config_filename):
        self.config_filename = config_filename
        self.read_config(config_filename)

    def read_config_optional(self, conf, parameter, default, dtype):
        try:
            self.config[parameter] = dtype(conf[parameter])
        except KeyError:
            self.config[parameter]=default
        
    def read_config(self, config_filename):
        ''' read a config file

        This method reads a config file (ini-style) and assigns the relevant parameters to self.config

        Parameters
        ----------
        config_filename : string
            name of config file
        '''
        conf = configobj.ConfigObj(config_filename)
        if not conf:
            self.write_default_config(config_filename)
            raise IOError("Could not find the config file '%s'.\nI wrote one with default values. Please edit and run again."%(config_filename))
        self.config = dict(datadir=conf['datadir'],
                           outputfilename=conf['outputfilename'],
                           roll_mean = float(conf['roll_mean']),
                           dvl_roll_offset = float(conf['dvl_roll_offset']),
                           att_time_offset = float(conf['att_time_offset']),
                           n_files = int(conf['n_files']),
                           start_file = int(conf['start_file']),
                           glider = conf['glider'])
        self.read_config_optional(conf, 'water_density', 1025, float)
        self.read_config_optional(conf, 'water_salinity', 35, float)
        self.read_config_optional(conf, 'pd0_extension', 'pd0', str)
        self.read_config_optional(conf, 'fit', 'linear', str)
        self.read_config_optional(conf, 'coord', 'BEAM', str)
        self.read_config_optional(conf, 'pitch_correction_factor', 1, float)
        self.read_config_optional(conf, 'pitch_correction_offset', 0, float)
            
    def write_default_config(self, config_filename):
        ''' write a default config file

        This method writes a default config file. The values in it are almost certainly wrong
        and have to be edited first.

        Parameters
        ----------
        config_filename : string
            name of configuration file
        '''
        conf = dict()
        conf['glider'] = 'unknown'
        conf['datadir'] = os.getcwd()
        conf['outputfilename']='noname.ndf'
        conf['roll_mean'] = 0.0
        conf['dvl_roll_offset'] = 0.0
        conf['att_time_offset'] = 0.0
        conf['n_files'] = 10
        conf['start_file'] = 0
        conf['water_density'] = 1025
        conf['date'] = 'today'
        conf['pd0_extension'] = 'pd0'
        conf['coord'] = 'BEAM'
        conf['pitch_correction_factor']=1.0
        conf['pitch_correction_offset']=0.0
        self.write_config(config_filename, conf)

    def write_config(self, config_filename, config):
        ''' write a default config file

        Parameters
        ----------
        config_filename : string
            name of configuration file
        config : dictionary
            configuration parameters
        '''
        cnf = configobj.ConfigObj()
        cnf.filename=config_filename
        for k, v in config.items():
            cnf[k]=v
        cnf.write()
        
        


class DVL_GliderPitch_Tune(DVL_GliderPitchBase):
    def __init__(self, config_filename):
        super().__init__(config_filename)
        
    def make_pipeline(self):
        '''Makes a pipeline

        This method sets up a pipeline which filters bad bottom track values, and performs
        a rotation to the SFU coordinate frame after applying some offsets in roll
        '''

        roll_offset = -self.config['roll_mean'] + self.config['dvl_roll_offset']
        
        bottomtrack_filter = rdi_qc.ValueLimit(drop_masked_ensembles=False)
        bottomtrack_filter.set_discard_condition('bottom_track', 'PG4','<', 50)
        bottomtrack_filter.set_discard_condition('bottom_track', 'BTVel1','||>', 0.75)
        bottomtrack_filter.set_discard_condition('bottom_track', 'BTVel2','||>', 0.75)
        bottomtrack_filter.set_discard_condition('bottom_track', 'BTVel3','||>', 0.75)
        bottomtrack_filter.set_discard_condition('bottom_track', 'Range3','>', 60)
        if self.config['coord'] == 'ENU':
            # note: the order of multiplication is reversed wrt the sequence of applying...
            transform = rdi_transforms.TransformXYZ_SFU(0, self.MOUNT_PITCH, roll_offset)            
            transform @= rdi_transforms.TransformSFU_XYZ(0, self.MOUNT_PITCH, 0)
            transform @= rdi_transforms.TransformENU_SFU()
        elif self.config['coord'] == 'BEAM':
            # note: the order of multiplication is reversed wrt the sequence of applying...
            transform = rdi_transforms.TransformXYZ_SFU(0, self.MOUNT_PITCH, roll_offset)
            transform @= rdi_transforms.TransformBEAM_XYZ()
        else:
            raise ValueError('Unhandled case.')
        pipeline = rdi_reader.make_pipeline(transform, bottomtrack_filter)
        return pipeline
    
        
    def __cost_fun(self, x, pitch, by, bz, wg):
        a, b = x
        wgp = -np.sin(pitch*a+b)*by -np.cos(pitch*a+b)*bz
        return np.linalg.norm(wg-wgp)

    def __cost_fun_proportional(self, x, pitch, by, bz, wg, b):
        a = x
        wgp = -np.sin(pitch*a+b)*by -np.cos(pitch*a+b)*bz
        return np.linalg.norm(wg-wgp)

    def get_depth_rate(self, data, gld_fns):
        ''' get depth-rate and pitch from glider data
        
        This method computes the depth rate from the pressure sensor and reads the pitch
    
        Parameters
        ----------
        data : rdi_writer.DataStructure
        gld_fns : list of strings
            list of glider binary filenames
        Returns
        -------
        depth-rate, pitch : (array, array)
        '''
        t_offset = self.config['att_time_offset']
        t = data.Time + t_offset

        dbd = dbdreader.MultiDBD(filenames = gld_fns, include_paired = True)
        #tg, m_depth_rate = dbd.get("m_depth_rate")
        tmp = dbd.get_sync("sci_ctd41cp_timestamp", ["sci_water_pressure", "m_pitch"])
        _, tg, P, pitch = tmp.compress(tmp[1]>1e8, axis=1)
        
        tmp = dbd.get_sync("m_water_pressure", ["m_pitch"])
        tg, P, pitch = tmp
        
        m_depth_rate = np.gradient(P*1e5/self.config['water_density']/9.81)/np.gradient(tg)
        #m_depth_rate = np.convolve(m_depth_rate, np.ones(5)/5, 'same')
        wg = np.interp(t, tg, -m_depth_rate)
        pitch_ = np.interp(t, tg, pitch)
        return wg, pitch_
    
    def fit_to_depth_rate(self, data, depth_rate):
        ''' Tune pitch angle scaling such that the vertical bottom track 
            velocity matches the depth rate

        Parameters
        ----------
        data : rdi_writer.DataStructure
            data read from the pipeline (sink)
        depth_rate : array
            depth-rate as observed by the glider

        Returns
        -------
        float, float
            scaling factor for pitch, pitch offset
        '''
        t = data.Time
        pitch = data.Pitch
        by = data.bottom_track_forward
        bz = data.bottom_track_up

        # optimise in two steps. First, using a good guess, and then
        # based on this one, just keep those values for which the
        # estimated value for depth_rate differs less than 3 cm/s compared to
        # the depth rate, to remove outliers.
        if self.config['fit']=='linear':
            idx = np.where(np.isfinite(by+bz+pitch+depth_rate))[0]
            a,b  = fmin(self.__cost_fun, [0.82, 0], args=(pitch[idx], by[idx], bz[idx], depth_rate[idx]))
            wgp = -np.sin(pitch*a+b)*by -np.cos(pitch*a+b)*bz
            idx = np.where(np.abs(depth_rate-wgp)<0.03)[0]
            a,b  = fmin(self.__cost_fun, [0.83, 0], args=(pitch[idx], by[idx], bz[idx], depth_rate[idx]))
        elif self.config['fit']=='proportional':
            b = self.config['pitch_correction_offset']
            idx = np.where(np.isfinite(by+bz+pitch+depth_rate))[0]
            a  = fmin(self.__cost_fun_proportional, 0.82,
                        args=(pitch[idx], by[idx], bz[idx], depth_rate[idx], b))
            wgp = -np.sin(pitch*a+b)*by -np.cos(pitch*a+b)*bz
            idx = np.where(np.abs(depth_rate-wgp)<0.03)[0]
            a  = fmin(self.__cost_fun_proportional, 0.83, args=(pitch[idx], by[idx], bz[idx], depth_rate[idx],b))
            a=float(a)
        return a, b

   
    def process(self):
        ''' Process the data

        Based on the settings from the config file, a selection of PD0 files
        are processed, and pitch coefficients are tuned.

        Returns
        -------
        float, float
            scaling factor for pitch, pitch offset
        '''
        dvl_fns, gld_fns = get_filenames(self.config['datadir'], self.config['glider'],
                                         n_files = self.config['n_files'], start = self.config['start_file'],
                                         pd0_extension=self.config['pd0_extension'])
        pipeline = self.make_pipeline()
        reader = rdi_reader.PD0()
        data_structure = rdi_writer.DataStructure()
        data_structure.add_parameter_list("bottom_track", "Range1", "Range2", "Range3", "Range4")

        reader.send_to(pipeline)
        pipeline.send_to(data_structure)

        print("Going to process the DVL files (%d in total)"%(len(dvl_fns)))

        reader.process(dvl_fns)
        # correct DVL time for offset
        depth_rate, pitch = self.get_depth_rate(data_structure, gld_fns)
        a,b = self.fit_to_depth_rate(data_structure, depth_rate)
        self.config['pitch_correction_factor'] = a
        self.config['pitch_correction_offset'] = b
        self.write_config(self.config_filename, self.config)
        
        self.report(a,b)
        f, ax = self.make_graph(data_structure, depth_rate, pitch, a, b)
        self.data = data_structure
        self.f = f
        self.ax = ax
        return a,b
    
    def report(self, a, b):
        ''' Reports the scaling parameters to the screen

        Parameters
        ----------
        a : float
            scaling factor for pitch
        b : float
            offset for pitch
        '''
        print("-"*40)
        print("Correction factors for pitch and roll: %f"%(a))
        print("Pitch offset: %f (%f deg)"%(b, b*180./np.pi))
        print("-"*40)
        print()

    def make_graph(self, data, depth_rate, pitch, a, b):
        ''' Make a graph
        
        Makes a graph facilitating inspection of the results.
        
        Parameters
        ----------
        data : rdi_writer.DataStructure
            data read from the pipeline
        depth_rate : array
            depth-rate as observed by the glider
        pitch : array
            pitch measured by the glider
        a : float
            scaling factor pitch
        b : float
            offset pitch
        
        Returns
        -------
        figure, axis

        '''
        t_offset = self.config['att_time_offset']
        t = data.Time
        pitch_c = data.Pitch*a + b
        by = data.bottom_track_forward
        bz = data.bottom_track_up
        wgp = -np.sin(pitch_c)*by -np.cos(pitch_c)*bz
        wg0 = -np.sin(data.Pitch)*by -np.cos(data.Pitch)*bz
        f, ax = pyplot.subplots(2,1, sharex=True)
        ax[0].plot(dates.epoch2num(t), depth_rate, label='m_depth_rate')
        ax[0].plot(dates.epoch2num(t), wgp, label='vertical bottom track')
        #ax[0].plot(dates.epoch2num(t), wg0, label='vertical bottom track raw')
        ax[0].set_ylabel('Vertical velocity (m/s)')
        ax[0].legend()
        ax[1].plot(dates.epoch2num(t-t_offset), pitch_c, 'C0', alpha=0.1, label='DVL (no offset)')
        ax[1].plot(dates.epoch2num(t), pitch_c, 'C1',label='DVL (offset=%.2f (s))'%(t_offset))
        ax[1].plot(dates.epoch2num(t), pitch*a+b, 'C2',label='Glider')
        ax[1].legend()
        ax[1].xaxis_date()
        return f, ax


class DVL_GliderPitch(DVL_GliderPitchBase):
    ''' Processes the data by applying a number of corrections and filters.
    
    * corrects for speed of sound, given the salinity set in the config file
    * removes unrealistic velocities
    * applies pitch correction
    * applies a time shift

    Parameters
    ----------
    config_filename : string
        name of configuration file
    a : float
        scaling factor for pitch
    b : float
        offset for pitch
    '''
    def __init__(self, config_filename, a, b):
        super().__init__(config_filename)
        self.a = a
        self.b = b

    def setup_pipeline(self):
        ''' Sets up an elaborate list of filters
        '''
        roll_offset = -self.config['roll_mean'] + self.config['dvl_roll_offset']

        adv_att = rdi_corrections.AdvanceAttitudeAngles(-self.config['att_time_offset'], window_length=10)
        
        # note: the order of multiplication is reversed wrt the sequence of applying...
        # brings to Beam coordinates
        transform0 = rdi_transforms.TransformXYZ_BEAM()
        transform0 *= rdi_transforms.TransformSFU_XYZ(0, self.MOUNT_PITCH, 0)
        transform0 *= rdi_transforms.TransformENU_SFU()

        # brings to SFU coordinates
        transform1 = rdi_transforms.TransformXYZ_SFU(0, self.MOUNT_PITCH, roll_offset)
        transform1 *= rdi_transforms.TransformBEAM_XYZ()

        #brings to ENU coordinates
        transform2 = rdi_transforms.TransformSFU_ENU()

        # Setup the attitude correction
        att_cor = rdi_corrections.AttitudeCorrectionTiltCorrection(tilt_correction_factor=self.a,
                                                                   pitch_offset = self.b, method='rotation')

        # Quality control filters:

        # filter nonsensical velocities
        max_velocity = 0.75
        qc_u_limit = rdi_qc.ValueLimit()
        qc_u_limit.set_discard_condition('velocity','Velocity1','||>',max_velocity)
        qc_u_limit.set_discard_condition('velocity','Velocity2','||>',max_velocity)
        qc_u_limit.set_discard_condition('velocity','Velocity3','||>',max_velocity)
        qc_u_limit.set_discard_condition('velocity','Velocity4','||>',max_velocity)
        qc_u_limit.set_discard_condition('bottom_track','BTVel1','||>',max_velocity)
        qc_u_limit.set_discard_condition('bottom_track','BTVel2','||>',max_velocity)
        qc_u_limit.set_discard_condition('bottom_track','BTVel3','||>',max_velocity)
        qc_u_limit.set_discard_condition('bottom_track','BTVel4','||>',max_velocity)

        # filter too noisy data
        qc_snr_limit = rdi_qc.SNRLimit(3)

        # filter too strong echos
        qc_amplitude_limit = rdi_qc.AcousticAmplitudeLimit(75)


        # Speed of sound correction
        c = rdi_corrections.SpeedOfSoundCorrection()

        self.pipeline.add(adv_att)
        self.pipeline.add(qc_u_limit)
        self.pipeline.add(qc_snr_limit)
        self.pipeline.add(qc_amplitude_limit)
        self.pipeline.add(transform0)
        self.pipeline.add(lambda g:
                          c.current_correction_at_transducer_from_salinity(g, self.config['water_salinity']))
        self.pipeline.add(transform1)
        self.pipeline.add(att_cor)
        self.pipeline.add(transform2)
    
    def process(self):
        ''' Process the PD0 files as given in the configuration file
        '''
        dvl_fns, gld_fns = get_filenames(self.config['datadir'], self.config['glider'],
                                         pd0_extension=self.config['pd0_extension'])

        self.setup_pipeline()
        writer = rdi_writer.NDFWriter()
        writer.output_file = self.config['outputfilename']

        writer(self.pipeline(dvl_fns))
        


