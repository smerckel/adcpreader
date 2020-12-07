from functools import lru_cache
import logging

import netCDF4
import numpy as np
from scipy.signal import butter, filtfilt
import dbdreader

from slocum import ladcp
from profiles import iterprofiles 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)





class DVLProcessor(object):
    '''Class to process DVL data collected by a Slocum glider

    The class requires data from three different sources:

    * a NetCDF file with DVL data, processed by the rdi package
    * a NetCDF file with calibrated glider flight data (using deploymentcalibrator package)
    * dbd files for additional engineering data, such as gps coordinates.

    The glider data are split in profiles (most likely consisting of down casts only due to 
    the fact that DVL on the up cast make not so much sense). For each profile a single velocity
    profile is computed using the LoweredADCP approach. Which constraints are used are user-configurable.
    Valid options are:
    
    use_bottom_track
    use_barotropic_velocity_constraint
    use_surface_velocity_constraint

    Additionally gliderflight information can be used set, by the parameter
    use_glider_flight. Since this is still a relative measure, this is not sufficient
    to constrain the velocity profiles.

    Parameters
    ----------
    dvl_data_file: str
        filename of netCDF file with DVL data
    gld_flght_data_file : str
        filename of netCDF file with glider flight data
    dbd_pattern : str
        glob pattern matching the required dbd files. (ebd files are not required).
    rho_reference : float (1025)
        reference density, used to calculate bar-to-meter conversion
    buoyancy_sensor_name : str, {"m_de_oil_vol", "m_ballast_pumped"}
        name of the sensor indicating the buoyancy drive of the glider
    filter_period : float (15)
        Sets the cut-off frequency of a low pass filter, used to smooth the pressure readings 
        in order to remove any effects in pressure anomalies due to surface waves
    
    Aditional Parameters
    --------------------
    **dbd_kwds : further keywords can be prescribed which are passed on to the dbdreader.MultiDBD class.


    Example
    -------

    >>> dvl_data_file = "/home/lucas/gliderdata/nsb3_201907/process/dvl_3beam_nsb3_201907.nc"
    >>> gld_flght_data_file = "/home/lucas/gliderdata/nsb3_201907/process/comet_nsb3_201907.nc"
    >>> dbd_pattern = "/home/lucas/gliderdata/nsb3_201907/hd/comet*.dbd"


    >>> dvlp = DVLProcessor(dvl_data_file, gld_flght_data_file, dbd_pattern,
                        rho_reference=1025, buoyancy_sensor_name='m_ballast_pumped',
                        filter_period=15)


    >>> ladcp_configuration = dict(use_bottom_track=True,
                               use_barotropic_velocity_constraint=False,
                               use_surface_velocity_constraint=False,
                               use_glider_flight=False,
                               z_min=5,
                               z_max=45,
                               dz=0.5)

    >>> dvlp.configure_ladcp(**ladcp_configuration)

    >>> sv, bv = dvlp.compute_surface_and_barotropic_water_velocities()
    >>> velocity_data = dvlp.compute_velocity_matrix()
    
    '''
    def __init__(self, dvl_data_file, gld_flght_data_file, dbd_pattern,
                 magnetic_variation = 0.,
                 rho_reference=1025,
                 buoyancy_sensor_name="m_ballast_pumped",
                 filter_period=15,
                 **dbd_kwds):
        self.dvl_dataset = netCDF4.Dataset(dvl_data_file, mode='r')
        self.gld_flght_dataset = netCDF4.Dataset(gld_flght_data_file, mode='r')
        self.dbd = dbdreader.MultiDBD(pattern=dbd_pattern, **dbd_kwds)
        self.bar_to_metre = 10**5/9.81/rho_reference
        self.altitude_correction_factor = 1.01 # see ~/working/git/gliderflight_waves_paper/wave_effects/vertical_motion_from_dvl.py
        self.buoyancy_sensor_name = buoyancy_sensor_name
        self.filter_period = filter_period
        self.magnetic_variation = magnetic_variation
        self.ladcp = ladcp.LoweredADCP()


    def configure_ladcp(self, **kwds):
        '''
        Configuration of the loweredADCP method.
        
        Parameters
        ----------
        **kwds can be any of

        use_bottom_track : bool
            use bottom track to constrain profile (True),
        use_barotropic_velocity_constraint : bool
            use barotropic velocity as constraint (True)
        use_surface_velocity_constraint : bool
            use surface (drift) velocity as constraint (True)
        use_glider_flight : bool
            use glider flight information (True)


        z_min : float
            minimum depth for computing velocity data
        z_max : float
            maximum depth for computing velocity data
        dz : float
            grid size

        Returns
        -------
        dict:
            unprocessed or recognised keywords
        '''
        remaining_kwds = self.ladcp.method_settings(**kwds)
        remaining_kwds = self.ladcp.grid_settings(**remaining_kwds)
        return remaining_kwds
    
    @lru_cache()
    def compute_surface_and_barotropic_water_velocities(self):
        logger.info("Computing surface and barotropic velocities from glider data.")
        _, gf_data, surface_data = self.read_data()
        water_velocities = ladcp.WaterVelocities(gf_data, surface_data, self.magnetic_variation)
        sv = water_velocities.compute_surface_velocities()
        bv = water_velocities.compute_barotropic_velocities()
        return sv, bv

    def compute_velocity_matrix(self):
        logger.info("Computing velocity matrix.")
        dvl_data, gf_data, surface_data = self.read_data()
        if self.ladcp.use_surface_velocity_constraint or self.ladcp.use_barotropic_velocity_constraint:
            sv, bv = self.compute_surface_and_barotropic_water_velocities()

        
        self.ladcp.initialise_grid(r=dvl_data['r'])

        ps = iterprofiles.ProfileSplitter(data=dvl_data)
        ps.split_profiles(interpolate=True)
        n_ps = len(ps)
        
        water_velocity = np.ma.zeros((2, self.ladcp.zi.shape[0], n_ps), float)
        water_depths = np.ma.zeros(n_ps, float)
        profile_timestamps = np.zeros_like(water_depths)
        for i, p in enumerate(ps):
            logger.info(f"Profile {i+1} from {len(ps)}")
            profile_timestamps[i] = p.t_down
            _, pressure = p.get_downcast("pressure")
            _, altitude = p.get_downcast("altitude")
            ma_altitude = np.ma.masked_array(altitude, pressure*self.bar_to_metre<15)
            water_depth = self.ladcp.compute_waterdepth(ma_altitude, pressure*self.bar_to_metre)
            water_depths[i] = water_depth
            z = pressure*self.bar_to_metre
            for j, s in enumerate('u v'.split()):
                u = self._compute_velocity_profile(p, s, z, water_depth)
                _, water_velocity[j,:, i] = u
        return ladcp.Velocity_Data(profile_timestamps, self.ladcp.zi, *water_velocity, water_depths)
    
    def _compute_velocity_profile(self, p, s, z, water_depth, u_gf=0, u_barotropic=0, u_surface=0):
        _, u = p.get_downcast(s)
        _, bt_u = p.get_downcast(f'bt_{s}')
        # ensure bt_u is properly masked...
        bt_u.mask = np.isnan(bt_u.data)
        U = self.ladcp.compute_depth_referenced_velocity_matrix(u, z, waterdepth=water_depth)
        ug, uw = self.ladcp.compute_velocity_profile(U, bt_u, z, u_gf, 
                                                     u_barotropic, u_surface)
        return ug, uw

    @lru_cache()
    def read_data(self):
        logger.info("Reading dvl data...")
        dvl_data = self._read_dvl_data()
        logger.info("Reading dbd data...")        
        surface_data = self._read_dbd_data()
        logger.info("Reading glider flight data...")
        gf_data = self._read_gf_data()
        logger.info("Data read.")
        # add depth of glider to dvl_data:
        dvl_data['z'] = np.interp(dvl_data['time'], gf_data['time'], gf_data['pressure']*self.bar_to_metre)
        dvl_data['pressure'] = dvl_data['z']/self.bar_to_metre
        return dvl_data, gf_data, surface_data
        
    def _read_dvl_data(self):

        dvl_t = self.dvl_dataset.variables['time'][:].data # no mask 
        dvl_r = self.dvl_dataset.variables['z'][:].data    # no mask 
        dvl_u = self.dvl_dataset.variables['Velocity1'][:] 
        dvl_v = self.dvl_dataset.variables['Velocity2'][:] 
        dvl_w = self.dvl_dataset.variables['Velocity3'][:]
        dvl_bt_u = self.dvl_dataset.variables['BTVel1'][:] 
        dvl_bt_v = self.dvl_dataset.variables['BTVel2'][:] 
        dvl_bt_w = self.dvl_dataset.variables['BTVel3'][:]
        R1 = self.dvl_dataset.variables['Range1'][:]
        R3 = self.dvl_dataset.variables['Range3'][:]
        R = 0.5*(R1 + R3)*self.altitude_correction_factor
        dvl_data = dict(time=dvl_t, r=dvl_r, u=dvl_u, v=dvl_v, w=dvl_w,
                        bt_u=dvl_bt_u, bt_v=dvl_bt_v, bt_w=dvl_bt_w, altitude = R)
        return dvl_data

    def _read_dbd_data(self):
        t_gps, lat_gps, lon_gps, buoyancy_change = self.dbd.get_sync("m_gps_lat", "m_gps_lon", self.buoyancy_sensor_name) 
        surface_data = ladcp.Surface_Data(t_gps, lat_gps, lon_gps, buoyancy_change)
        return surface_data

    def _read_gf_data(self):
        t = self.gld_flght_dataset.variables['time'][:].data
        pressure_raw = 0.1 * self.gld_flght_dataset.variables['depth'][:].data
        pressure = self.low_pass_filter(t, pressure_raw, self.filter_period)
        u = self.gld_flght_dataset.variables['u'][:].data
        hdg = self.gld_flght_dataset.variables['heading'][:].data
        gf_U = self.gld_flght_dataset.variables['U'][:].data
        gf_u = np.cos(np.pi/2 - hdg*np.pi/180)
        gf_v = np.sin(np.pi/2 - hdg*np.pi/180)
        gf_data = dict(time=t, pressure=pressure, pressure_raw=pressure_raw,
                    gf_u=gf_u, gf_v=gf_v, gf_U=gf_U)
        return gf_data
    
    def low_pass_filter(self, t, x, T):
        dt = np.median(np.diff(t))
        ti = np.arange(t.min(), t.max()+dt, dt)
        xi = np.interp(ti, t, x)
        fn = 0.5*1/dt
        fc = 1/T
        b, a = butter(N=1, Wn=fc/fn, btype='low', analog=False, output='ba', fs=1/dt)
        xi = filtfilt(b, a, xi)
        return np.interp(t, ti, xi)

