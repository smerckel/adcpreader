from collections import namedtuple
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from profiles import iterprofiles
import latlonUTM

logger = logging.getLogger(name=__name__)

# Definition of some named tuples
Surface_Data = namedtuple('surface_data','t x y ballast'.split())
Surface_Velocity = namedtuple('surface_velocity', 't u v'.split())
Velocity_Data = namedtuple('velocity_data', 't z u v d'.split())
Glider_Velocity_Data = namedtuple('glider_velocity_data', 't u v'.split())

class WaterVelocities(object):
    ''' Class to compute water velocities from glider data using
        surface drift and dead reckoned trajectories.
    '''   
    def __init__(self, data, surface_data, magnetic_variation=None):
        ''' 
        Parameters
        ----------
        data : dictionary
               data dictionary passed to a ProfileSplitter instance. It therefore
               must contain the keys "time" and "pressure", as well as glider flight
               velocities "gf_u" and "gf_v" (eastward and northward velocities) and
               "gf_U", the incident water velocity.
               Other parameters may be present, but are not used.

        surface_data : Surface_Data named tuple, containing time (.t), gps_lat (.x) gps_lon (.y)
                       and concurrent buoyancy drive (.ballast) glider data

        magnetic_variation : float or None
               magnetic variation in DEGREES, correct the eastward and northward glider flight velocities
               for a magnetic variation
        '''
        self.ps = iterprofiles.ProfileSplitter(data)
        self.ps.split_profiles()
        self.surface_data = self._convert_gps_data(surface_data)
        self.magnetic_variation = magnetic_variation
        self.__velocities_integrated = False
        
    def integrate_glider_velocities(self):
        ''' Integrate the glider flight model velocities

        This method takes the data stored in the profilesplitter instance self.ps to integrate
        the flight model in time. The data are stored in the same data dictionary.
        '''
        tm = self.ps.data['time']
        u = self.ps.data['gf_u']
        v = self.ps.data['gf_v']
        if not self.magnetic_variation is None:
            u, v = self._correct_velocity_vectors_for_magnetic_variation(u, v)
        x = cumtrapz(u, tm, initial=0)
        y = cumtrapz(v, tm, initial=0)
        self.ps.data['x'] = x
        self.ps.data['y'] = y
        self.__velocities_integrated = True
        
    def compute_surface_velocities(self, ballast_min = 200, max_surface_time=20*60, min_gps_fixes=10):
        ''' Compute surface velocity from glider drift
        
        Parameters
        ----------
        ballast_min : float
                      if the buoyancy change is less than this threshold, the glider is assumed to have
                      started its dive; subsequent GPS fixes may be contanimated and are there for ignored.
        max_surface_time : float
                      surface intervals longer than this threshold are not considered as a normal interval;
                      no surface drift is calculated. Usually there is a problem, or a gap between deployments.
        min_gps_fixes : int
                      a surface interval must have at least this number of gps fixes, from which a drift is
                      to be calculated. A mathematical minimum is 2. If the glider reaches the surface during
                      a segment, it may pick up a handful of gps fixes. This may be useful to compute drift velocities
                      during the segment.

        Returns
        -------
        dictionary of
            time : array of floats
                   time stamps of surface velocity estimates
            u    : array of floats
                   eastward surface velocity
            v    : array of floats
                   norhtward surface velocity
        '''
        tm = self.ps.data['time']
        sd = self.surface_data
        surface_intervals = []
        n_segments = len(self.ps) - 1
        for cnt, (pleft, pright) in enumerate(zip(self.ps[:-1], self.ps[1:])):
            t0 = tm[pleft.i_up[-1]] # last timestamp of upcast
            t1 = tm[pright.i_down[0]] # first timestamp of downcast
            # compress all the data within these times and applying the buoyancy constraint:
            tmp = np.compress(np.logical_and(np.logical_and(sd.t >= t0,
                                                            sd.t <= t1),
                                             sd.ballast > ballast_min),
                              (sd.t, sd.x, sd.y, sd.ballast), axis=1)
            # only add to surface intervals if the other two constraints are met:
            if tmp.shape[1]>min_gps_fixes and tmp[0].ptp()<max_surface_time:
                surface_intervals.append(Surface_Data(*tmp))
            logger.info(f"Processing segment {cnt} of {n_segments}")
            logger.info(f"    surface time (min): {(t1-t0)/60}")
            logger.info(f"    gps available: {np.any(np.logical_and(sd.t >= t0,sd.t <= t1))}")
            
        self.surface_intervals = surface_intervals # the dead reckoning method needs this too
        # compute the drifting velocity
        us = []
        vs = []
        for s in surface_intervals:
            _us, _ = np.polyfit(s[0], s[1], 1)
            _vs, _ = np.polyfit(s[0], s[2], 1)
            us.append(_us)
            vs.append(_vs)
        us = np.array(us)
        vs = np.array(vs)
        ts = np.array([s[0].mean() for s in self.surface_intervals])
        self.surface_velocity = [Surface_Velocity(_ts, _us, _vs) for _ts, _us, _vs in zip(ts, us, vs)]
        return dict(time=ts, u=us, v=vs)
        
    def compute_barotropic_velocities(self, max_subsurface_time = 3*3600):
        ''' Compute depth and time averaged velocities per glider data segment
        
        Parameters
        ----------
        max_subsurface_time : float
               subsurface times in excess of this threshold are not considered.

        Returns
        -------
        dictionary of 
            time : array of floats
                 time stamps of barotropic velocity estimates
            u    : array of floats
                 eastward barotropic velocity
            v     : array of floats
                 norhtward barotropic velocity
        '''
        if not self.__velocities_integrated:
            self.integrate_glider_velocities()
        try:
            s0 = self.surface_intervals[:-1] # is computed in compute_surface_velocities()
        except AttributeError:
            raise ValueError("The method compute_surface_velocities() should be called first!")
        s1 = self.surface_intervals[1:]
        ub = []
        vb = []
        tb = []
        for _s0, _s1 in zip(s0, s1):
            if (_s1.t.min() - _s0.t.max())>max_subsurface_time:
                continue
            t_transect = 0.5 * (_s0.t.max() + _s1.t.min())
            tb.append(t_transect)
            _ub, _vb = self._deadreckon_transect(_s0, _s1)
            ub.append(_ub)
            vb.append(_vb)
        tb = np.array(tb)
        ub = np.array(ub)
        vb = np.array(vb)
        return dict(time=tb, u=ub, v=vb)

    def _deadreckon_transect(self, s0, s1):
        ''' compute the averaged water velocity for a transect
        
        Parameters
        ----------
        s0, s1 : surface_data named tuples
                 data of two sequential dive -> surfacing surface data tuples.

        Returns
        -------
        ub, vb : floats
                 velocity in eastward and northward direction, respectively, for this transect.
        '''
        t = self.ps.data['time']
        x = self.ps.data['x']
        y = self.ps.data['y']
        U = self.ps.data['gf_U']
        d = self.ps.data['pressure']*10
        # reduce to data matching this subsurface time only
        t, x, y, d, U = np.compress(np.logical_and(t>=s0.t.min(), t<=s1.t.max()), (t, x, y, d, U), axis=1)
        
        # start of the dive is first non zero U and ends where it is non zero too:
        idx = np.where(U>0)[0]
        t_dive = t[idx[0]]
        t_surface = t[idx[-1]]
        # guess where the glider really dived and surfaced:
        x_dive, y_dive = self._extrapolate(t_dive, s0)
        x_surface, y_surface = self._extrapolate(t_surface, s1)
        # match similation with gps derived positions, so that dives correspond:
        gf_x_dive = np.interp(t_dive, t, x)
        gf_y_dive = np.interp(t_dive, t, y)
        gf_x_surface = np.interp(t_surface, t, x)
        gf_y_surface = np.interp(t_surface, t, y)
        dx = x_dive - gf_x_dive
        dy = y_dive - gf_y_dive
        # drift:
        sx = x_surface - (gf_x_surface + dx)
        sy = y_surface - (gf_y_surface + dy)
        # and corresponding barotropic velocity:
        dt = t_surface-t_dive
        ub = sx/dt
        vb = sy/dt
        return ub, vb

    def _correct_velocity_vectors_for_magnetic_variation(self, u, v):
        ''' Correct any velocity readins from the model for magnetic variation
        
        Parameters
        ----------
        u, v : arrays of float
               eastward and northward velocities, respectively

        Returns
        -------
        Parameters
        ----------
        u, v : arrays of float
               corrected eastward and northward velocities, respectively
        '''
        # Rotates [u, v]T over an angle alpha. u and v are measured with respect to magnetic
        # north, but should be expressed with respect to true north.
        alpha = -np.pi*self.magnetic_variation/180.
        Up = (u + v *1j) * np.exp(alpha*1j)
        return Up.real, Up.imag
        
    def _convert_gps_data(self, surface_data):
        ''' Convert lat/lon to m easting and northing

        Parameters
        ----------
        surface_data : Surface_Data named tuple

        Converts latitude/longitude arrays into x and y coordinates, with reference to the 
        UTM tile of the first latitude/longitude pair.

        Returns
        -------
        surface_data : Surface_Data named tuple
                       the fields .x and .y now contain the coordinates in m
        '''
        x_gps, y_gps = latlonUTM.UTM(surface_data.x, surface_data.y)
        surface_data = Surface_Data(surface_data.t, x_gps, y_gps, surface_data.ballast)
        return surface_data
    

    def _extrapolate(self, t, s):
        ''' extrapolate surface position 
        Parameters
        ----------
        t : time to extrapolate position to
        s : surface data named tuple
        
        Returns
        -------
        x,y : floats
              extrapolated position
        '''
        x = self.__extrapolate(t, s.t, s.x)
        y = self.__extrapolate(t, s.t, s.y)
        return x, y
    
    def __extrapolate(self, T, t, x):
        a,b = np.polyfit(t, x, 1)
        X = a * T + b
        return X
        


class LoweredADCP(object):
    ''' A lowered ADCP method for computing velocities from a DVL carried by a glider.

    Similar to Visbeck (2009) and Todd et.al (2011).

    Additional feature: 
          
    This implementation allows to use glider flight data in improving the dvl data.

    '''
    def __init__(self):
        # set default settings:
        self.method_settings(use_bottom_track = True,
                             use_glider_flight = True,
                             use_barotropic_velocity_constraint = True,
                             use_surface_velocity_constraint = True)
        self.grid_settings(zi_min=5, zi_max=40, dz=1)
        self.ratio_limit = 1.2
        
    def initialise_grid(self, r):
        ''' Initialises the method

        Parameters
        ----------
        r : array of floats
            vector of acoustic bin distances 
        '''
        self.zi = np.arange(self.zi_min, self.zi_max, self.dz, float)
        # on interpolating outside the range, return index 0 for z<zmin, and N-1 for z>zi
        self.index_fun = interp1d(self.zi, np.arange(len(self.zi)),
                                  bounds_error=False, fill_value=(0, self.zi.shape[0]-1))
        self.dvl_r = r
        
    def method_settings(self, **kwds):
        ''' Specify the information used in trying to estimate the current profiles.

        Keywords can be any of:

        use_bottom_track : bool
            switch to use bottom track velocities (if available)
        use_glider_flight 
            switch to use glider flight velocities
        use_barotropic_velocity_constraint 
            switch to use barotropic (depth and time averaged velocities)
        use_surface_velocity_constraint
            switch to use surface water velocities
        '''
        valid_kwds = 'use_bottom_track use_glider_flight use_barotropic_velocity_constraint use_surface_velocity_constraint'.split()
        return self._process_keywords(valid_kwds, **kwds)
    
    def grid_settings(self, **kwds):
        ''' Specify the settings for the output grid

        Keywords can be any of:

        zi_min : float
             min water depth 
        zi_max : float
             max water depth
        dz : float
            vertical resolution 
        '''

        valid_kwds = 'zi_min zi_max dz'.split()
        return self._process_keywords(valid_kwds, **kwds)

    def _process_keywords(self, valid_kwds, **kwds):
        unused_keywords = {}
        for k, v in kwds.items():
            if k in valid_kwds:
                self.__dict__[k] = v
            else:
                unused_keywords[k] = v
        return unused_keywords

    
    def compute_velocity_profile(self, U, dvl_bt_u, dvl_gf_z, dvl_gf_u,
                                 u_barotropic=0., u_surface=0.):
        '''
        Computes the velocity profile
        
        Parameters
        ----------
        U : N X M array of floats
            DVL water velocities (relative to the glider), depth-referenced.
        dvl_bt_u : N array of floats
            DVL bottom track velocities
        dvl_gf_u : N array of floats
            Glider flight computed velocity component
        dvl_gf_z : N array of floats
            Depth of glider (most likely from glider flight model, but not necessarily so)
        u_barotropic : N array of floats or float
            barotropic velocity measurements
        u_surface : N array of floats or float
            surface velocity measurements

        Returns
        -------
        ug : N array of floats
            absolute glider velocity estimates
        uw : K array of floats
            absolute water velocity component

        N : number of ensembles in this profile
        M : number of acoustic bins
        K : length of depth vector, computed from zi_min, zi_max in dz steps.
        '''
        G, d = self.compute_G_and_d(U, dvl_bt_u, dvl_gf_z, dvl_gf_u,
                                    u_barotropic = u_barotropic, u_surface=u_surface)
        self.G = G # diagnostic
        self.d = d
        #self.U = U # diagnostic
        G, non_zero_columns = self.remove_zero_columns(G)
        self.Gp = G
        #
        nz = self.zi.shape[0]
        N = U.shape[0]
        NG = G.shape[0]
        if NG/N < self.ratio_limit:
            # all U are zero, no system matrix
            uw = np.ma.masked_all(nz)
            ug = np.ma.masked_all(0)
            logger.warning("Failed")
            return ug, uw

        # solve the system
        try:
            mp = np.linalg.inv(G.T @ G) @ G.T @ d
        except:
            uw = np.ma.masked_all(nz)
            ug = np.ma.masked_all(0)
            logger.warning("Failed inversion")
            return ug, uw

        m = np.ma.masked_all(N+nz, float)
        m[non_zero_columns] = mp[:,0]
        ug = m[:N]
        uw = m[N:]
        return ug, uw
    
    def compute_depth_referenced_velocity_matrix(self, dvl_u, dvl_gf_z, waterdepth=None, bottom_clearance=1):
        ''' Compute depth referenced velocity matrix

        The matrix with velocities recorded by the DVL are all relative to the glider's depth. This
        method constructs a velocity matrix with the first dimension corresponding to the depth a

        Parameters
        ----------
        dvl_u : N X M array of floats
            DVL water velocities (relative to the glider)

        dvl_gf_z : N array of floats
            Depth of glider (most likely from glider flight model, but not necessarily so)
       
        waterdepth: float or None
            Estimate of the waterdepth

        bottom_clearance : float
            Height above the bed where velocity readings should be discarded.

        Returns
        -------
        U: NxK array of floats
            Depth reference mapping of dvl measured velocities.

        N : number of ensembles in this profile
        K : length of depth vector, computed from zi_min, zi_max in dz steps.
        '''
        U = np.nan * np.zeros((dvl_gf_z.shape[0], self.zi.shape[0]), float)
        for i, _z in enumerate(dvl_gf_z):
            try:
                _u, _r = np.compress(~dvl_u[i].mask, (dvl_u[i].data, self.dvl_r), axis=1)
            except ValueError:
                # assume that dvl_u[i].mask is NOT an array:
                if dvl_u[i].mask: # all masked, don't use these data
                    continue
                else:
                    _u = dvl_u[i].data # all fine, use them all
                    _r = self.dvl_r
            if np.isfinite(waterdepth):
                _u, _r = np.compress(_r+_z < waterdepth-bottom_clearance, (_u, _r), axis=1)
            try:
                U[i,:] = np.interp(self.zi, _r+_z, _u, left=np.nan, right=np.nan)
            except ValueError:
                pass
        U = np.ma.masked_where(np.isnan(U), U)
        return U
    
    def compute_waterdepth(self, altitude, z):
        valid_readings = (altitude + z).compressed()
        if valid_readings.shape[0]:
            waterdepth = np.median(valid_readings)
        else:
            waterdepth = np.nan
        return waterdepth
        
    def compute_G_and_d(self, U, dvl_bt_u, dvl_gf_z, dvl_gf_u, u_barotropic=0., u_surface=0):
        ''' Computes G and d matrix/vector from measurments.

        Not to be called directly.

        Returns
        -------
        G: 2D array of floats
           system matrix
        d: vector of floats
           observation vector
        '''
        G = []
        d = []
        N = U.shape[0]
        nz = len(self.zi)
        g_right_filled = np.zeros(nz, int)
        # construct the G matrix

        for i, u in enumerate(U):
            if np.all(u==0):
                continue
            #idx = np.where(np.isfinite(u))[0]
            idx = np.where(~u.mask)[0] # perhaps exclude measurements that are *exactly* zero?
            # in this ping we have len(idx) measurements. So G get this number of lines.
            for j in idx:
                g_left = np.zeros(N, float)
                g_left[i]=-1.
                g_right = np.zeros(nz, float)
                g_right[j] = 1
                g_right_filled[j]=1 # to keep track zero columns later.
                G.append(np.hstack((g_left, g_right)))
                d.append(u[j])
                
            # add bottom track information, if available
            if not np.ma.is_masked(dvl_bt_u[i]) and self.use_bottom_track:
                g_left = np.zeros(N, float)
                g_left[i]=1.
                g_right = np.zeros(nz, float)
                G.append(np.hstack((g_left, g_right)))
                d.append(-dvl_bt_u[i])

            # adding glider flight information
            if self.use_glider_flight:
                g_left = np.zeros(N, float)
                g_left[i]=1.
                g_right = np.zeros(nz, float)
                k = (self.index_fun(dvl_gf_z[i])+0.5).astype(int)
                g_right[k] = -1
                g_right_filled[k]=1 # to keep track zero columns later.
                G.append(np.hstack((g_left, g_right)))
                d.append(dvl_gf_u[i])
                
            # I think the code below should not be here. Only one surface measurement should be used?    
            #if self.use_surface_velocity_constraint and np.isfinite(u_surface) and dvl_gf_z[i]<1:
            #    g_left = np.zeros(N, float)
            #    g_right = np.zeros(nz, float)
            #    g_left[i] = 1
            #    G.append(np.hstack((g_left, g_right)))
            #    d.append(u_surface)

                
        # adding surface velocity
        if self.use_surface_velocity_constraint and np.isfinite(u_surface):
            g_left = np.zeros(N, float)
            g_right = np.zeros(nz, float)
            g_right[0] = 1
            g_right_filled[0]=1 # to keep track zero columns later.
            G.append(np.hstack((g_left, g_right)))
            d.append(u_surface)


            
        # adding constraint by adding all velocities.
        if self.use_barotropic_velocity_constraint and np.isfinite(u_barotropic):
            g_left = np.zeros(N, float)
            ## g_right is now equal to g_right_filled.
            G.append(np.hstack((g_left, g_right_filled)))
            d.append(u_barotropic * g_right_filled.sum())
        G = np.array(G)
        d = np.array(d).reshape(-1, 1)
        return G, d

    def remove_zero_columns(self, G):
        ''' Remove zero filled columns
        
        Parameters
        ----------
        G: 2D array

        Returns
        -------
        Gp : 2D array
            As G, but reduced by removing zero columns
        non_zero_columns : list
            list with index numbers containing what columns of the original matrix were retained.
        '''
        non_zero_columns = []
        Gp = []
        for i, g in enumerate(G.T):
            if not np.ma.allclose(g, 0, masked_equal=True):
                Gp.append(g)
                non_zero_columns.append(i)

        Gp = np.array(Gp).T
        non_zero_columns = np.array(non_zero_columns)
        return Gp, non_zero_columns

