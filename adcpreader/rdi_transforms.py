'''

A module to perform rotation transformations on

* velocity vectors 1-4
* bottom_track values 

per ensemble

Typical use is to set up an ensemble generator and a pipeline of transformations

Examples
--------
    >>> bindata = pd0.PD0()
    >>> ensembles = bindata.ensemble_generator(filenames)

    >>> t1 = TransformENU_SFU()
    >>> t2 = TransformSFU_XYZ(hdg=0, pitch=0.1919, roll=0)
    >>> t3 = TransformXYZ_SFU(hdg=0, pitch=0.2239, roll=0.05)

    >>> t4 = t3*t2*t1

    >>> ensembles = t4(ensembles)

    >>> for ens in ensembles :
    >>>     print(ens)

'''

from collections import namedtuple
from itertools import chain

import numpy as np

from adcpreader import __VERSION__
from adcpreader.coroutine import coroutine, Coroutine

Attitude = namedtuple('Attitude', 'hdg pitch roll')
Beamconfig = namedtuple('Beamconfig', 'a b c d facing')

VELOCITY_FIELDS = dict(Beam=[f'To Beam {i+1}' for i in range(4)],
                       Instrument=['To X', 'To Y', 'To Z', 'To error'],
                       Ship=['To Starboard', 'To Bow', 'To Mast', 'To error'],
                       Earth=['To East', 'To North', 'To Up', 'To error'])
FOUR_BEAM_SOLUTION=0
THREE_BEAM_SOLUTION=1
THREE_BEAM_SOLUTION_DISCARD_FOURTH=2
THREE_BEAM_SOLUTION_DISCARD_THIRD=3


class RotationMatrix(object):
    ''' A rotation matrix class as defined in the RDI manual '''
    def create_matrix(self, heading, pitch, roll):
        CH = np.cos(heading) 
        CP = np.cos(pitch)   
        CR = np.cos(roll)    
        SH = np.sin(heading) 
        SP = np.sin(pitch)   
        SR = np.sin(roll)    
        M = np.array([[CH*CR+SH*SP*SR, SH*CP, CH*SR-SH*SP*CR, 0],
                      [-SH*CR+CH*SP*SR, CH*CP,-SH*SR-CH*SP*CR,0],
                      [-CP*SR, SP, CP*CR, 0],
                      [0, 0 ,0, 1]])
        return M
        
    def __call__(self, heading, pitch, roll):
        return self.create_matrix(heading, pitch, roll)

class TransformMatrix(object):
    def __init__(self, use_beam_solution=0):
        self.beam_solution = use_beam_solution
        
    def create_matrix(self, a, b, c, d):
        if self.beam_solution==FOUR_BEAM_SOLUTION: # use all four beams:
            M = np.array([[c*a, -c*a, 0, 0],
                          [0  ,    0, -c*a, c*a],
                          [b  ,    b,    b,   b],
                          [d  ,    d,   -d,  -d]])
        elif self.beam_solution==THREE_BEAM_SOLUTION_DISCARD_FOURTH: # use beams 1,2, 3 and leave out 4
            M = np.array([[c*a, -c*a, 0, 0],
                          [c*a  ,   c*a, -2*c*a, 0],
                          [2* b  ,    2*b,    0,   0],
                          [0  ,   0,   0,  0]])
        elif self.beam_solution==THREE_BEAM_SOLUTION_DISCARD_THIRD: # use beams 1,2, 4 and leave out 3
            M = np.array([[c*a, -c*a, 0, 0], 
                          [-c*a  ,  - c*a, 0, 2*c*a],
                          [2* b  ,    2*b,    0,   0],
                          [0  ,   0,   0,  0]])
        elif self.beam_solution==THREE_BEAM_SOLUTION: # use beams 1, 2 and 3, and do proper projection
            theta = np.pi/180*30
            alpha = np.pi/180*15
            a11 = np.sin(theta)
            a12 = np.sin(alpha)*np.cos(theta)
            a13 = np.cos(alpha)*np.cos(theta)
            
            a21 = -np.sin(theta)
            a22 = np.sin(alpha)*np.cos(theta)
            a23 = np.cos(alpha)*np.cos(theta)

            a31 = 0
            a32 = -np.sin(theta)*np.cos(alpha) + np.sin(alpha)*np.cos(theta)
            a33 = np.sin(theta)*np.sin(alpha) + np.cos(alpha)*np.cos(theta)
            
            A = np.array([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])
            Ainv = np.linalg.inv(A)
            M = np.zeros((4,4),float)
            M[:3, :3] = Ainv
        else:
            raise NotImplementedError("Beam solution matrix not implemented yet.")
        return M
    
    def __call__(self, a, b, c, d):
        return self.create_matrix(a, b, c, d)


    
class Transform(Coroutine):
    ''' Base transform class.

    Implements the transformations and multiplication, but not the definition of the rotation matrix.

    This class should be subclassed.
    '''
    # which parameters should be transformed
    PARAMS = dict(velocity = ['Velocity'],
                  bottom_track = ['BTVel'])
    CACHE = {}
    
    def __init__(self, inverse = False):
        super().__init__()
        self.inverse = inverse
        self.coro_fun = self.coro_transform_ensembles()

    @coroutine
    def coro_transform_ensembles(self):
        ''' coroutine  transforming ensembles.'''
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                self.transform_velocities_in_ensemble(ens)
                self.send(ens)
        self.close_coroutine()

    def set_coordinate_systems(self, old_coordinate_system, new_coordinate_system, inverse=False):
        if not inverse:
            self.new_coordinate_system = new_coordinate_system
            self.old_coordinate_system = old_coordinate_system
        else:
            self.old_coordinate_system = new_coordinate_system
            self.new_coordinate_system = old_coordinate_system
            
        
    # provide support for new matrix multiplication notation @
    def __matmul__(self, ri):
        return self.__mul__(ri)
    
    def __mul__(self, ri):
        ''' Creates a create_transformation_matrix() method from the left and right arguments of the * operator. '''
        T = Transform()
        T.create_transformation_matrix = lambda *x: self.create_transformation_matrix(*x) @ ri.create_transformation_matrix(*x)
        if self.new_coordinate_system:
            T.new_coordinate_system = self.new_coordinate_system
            T.old_coordinate_system = ri.old_coordinate_system
        else:
            T.new_coordinate_system = ri.new_coordinate_system
            raise ValueError("Fix me: Don't know what to set here for the old coordinate system.")
        return T

    def get_beam_configuration(self, ens):
        try:
            a, b, c, d, facing = self.CACHE['beamconfig']
        except KeyError:
            fixed_leader = ens['fixed_leader']
            beam_angle, _ = fixed_leader['Beam_Angle'].split()
            beam_pattern = fixed_leader['Beam_Pattern']
            facing = fixed_leader['Xdcr_Facing']
            theta = float(beam_angle)*np.pi/180.
            a = 1/(2*np.sin(theta))
            b = 1/(4*np.cos(theta))
            c = int(beam_pattern=='Convex')*2-1
            d = a/np.sqrt(2)
            self.CACHE['beamconfig'] = a, b, c, d, facing
        return a, b, c, d, facing
    
    def get_params(self, ens):
        try:
            params = self.CACHE['params']
        except KeyError:
            ensemble_keys = list(ens.keys())
            params = dict([(k,v) for k, v in Transform.PARAMS.items() if k in ensemble_keys])
            self.CACHE['params'] = params
        return params

    def transform_velocities_in_ensemble(self, ens):
        ''' Transforms the velocities in given ensemble. 
        '''
        self.__check_coordinate_system(ens)
        hdg = ens['variable_leader']['Heading']*np.pi/180.
        pitch = ens['variable_leader']['Pitch']*np.pi/180.
        roll = ens['variable_leader']['Roll']*np.pi/180.
        attitude = Attitude(hdg, pitch, roll)

        a,b,c,d, facing = self.get_beam_configuration(ens)
        beamconfig = Beamconfig(a, b,c,d, facing)

        R = self.create_transformation_matrix(attitude, beamconfig)
        params = self.get_params(ens)
        self.__transform_velocities_in_ensemble(ens, R, params)
        self.update_coordinate_frame_setting(ens)
        self.post_modify_ensemble(ens)

    def update_coordinate_frame_setting(self, ens):
        ''' Writes the new coordinate frame setting and records the original setting. '''
        if self.new_coordinate_system:
            ens['fixed_leader']['OriginalCoordXfrm'] = ens['fixed_leader']['CoordXfrm']
            ens['fixed_leader']['CoordXfrm'] = self.new_coordinate_system
            for i, v in enumerate(VELOCITY_FIELDS[self.new_coordinate_system]):
                ens['fixed_leader'][f"Vel_field{i+1}"] = v
        else:
            raise ValueError('new_coordinate_system is NOT set!')

    def post_modify_ensemble(self, ens):
        pass
    
    def __check_coordinate_system(self, ens):
        if ens['fixed_leader']['CoordXfrm'] != self.old_coordinate_system:
            msg = '''Cannot apply the transformation as the transformation
matrix appears to transform from a different coordinate
system than the ensemble has.'''
            err_value = dict(ensemble = ens['fixed_leader']['CoordXfrm'],
                             transformation = self.old_coordinate_system)
            raise ValueError(msg, err_value)
        
            
    def __transform_velocities_in_ensemble(self, ens, R, params):
        for k, v in params.items():

            for _v in v:
                # make a note of the mask of this variable, if any.
                try:
                    #mask = ens[k]['%s%d'%(_v,1)].mask
                    mask = np.array([ens[k]['%s%d'%(_v,i+1)].mask.astype(float) for i in range(4)])
                except AttributeError:
                    mask = None
                except KeyError: # nothing to do
                    continue 
                x = np.array([ens[k]['%s%d'%(_v,i+1)] for i in range(4)])
                if x.shape[0] == 1: # for bottom track values
                    xp = np.array(R @ x.T)
                    for i in range(4):
                        ens[k]['%s%d'%(_v, i+1)] = float(xp[i])
                else:
                    xp = np.array(R @ x)
                    if mask is None: # no mask to apply
                        for i in range(4):
                            ens[k]['%s%d'%(_v, i+1)] = xp[i]
                    else: # apply (rotated mask)
                        maskp = np.array(R @ mask)!=0
                        for i in range(4):
                            ens[k]['%s%d'%(_v, i+1)] = np.ma.masked_array(xp[i], maskp[i])
                            
        
            
class TransformSFU_ENU(Transform):
    def __init__(self, inverse = False):
        super().__init__(inverse)
        self.set_coordinate_systems('Ship', 'Earth', inverse)

    def create_transformation_matrix(self, attitude, beamconfig):
        R = RotationMatrix()
        if self.inverse:
            return R(attitude.hdg, attitude.pitch, attitude.roll).T
        else:
            return R(attitude.hdg, attitude.pitch, attitude.roll)

class TransformXYZ_ENU(Transform):
    def __init__(self, inverse = False):
        super().__init__(inverse)
        self.set_coordinate_systems('Instrument', 'Earth', inverse)

    def create_transformation_matrix(self, attitude, beamconfig):
        R = RotationMatrix()
        # if upward looking, rotate 180 over S-axis 
        roll_adjustment = np.pi * int(beamconfig.facing == 'Up')

        if self.inverse:
            return R(attitude.hdg, attitude.pitch, attitude.roll+roll_adjustment).T
        else:
            return R(attitude.hdg, attitude.pitch, attitude.roll+roll_adjustment)


class TransformRotation(Transform):
    def __init__(self, hdg, pitch, roll, new_coordinate_system=None):
        inverse = False
        super().__init__(inverse)
        R = RotationMatrix()
        self.R = R(hdg, pitch, roll)
        self.new_coordinate_system = new_coordinate_system
            
    def create_transformation_matrix(self, *p):
        return self.R

class TransformBEAM_XYZ(Transform):
    def __init__(self,inverse = False, use_beam_solution=FOUR_BEAM_SOLUTION):
        super().__init__(inverse)
        self.set_coordinate_systems('Beam', 'Instrument', inverse)
        self.beam_solution = use_beam_solution
        
    def create_transformation_matrix(self, attitude, beamconfig):
        try:
            return self.R
        except AttributeError:
            R = TransformMatrix(self.beam_solution)
            self.R = R(beamconfig.a, beamconfig.b, beamconfig.c, beamconfig.d)
            if self.inverse:
                self.R = np.linalg.inv(self.R) # Transpose is not okay for non-rotational matrices.
            return self.R
        
    #overloaded method.    
    def post_modify_ensemble(self, ens):
        if self.beam_solution==THREE_BEAM_SOLUTION:
            ens['fixed_leader']['DepthCellSize']*=np.cos(15*np.pi/180)
            
class TransformXYZ_SFU(Transform):
    def __init__(self, hdg, pitch, roll, inverse = False):
        super().__init__(inverse)
        self.attitude = Attitude(hdg, pitch, roll)
        self.set_coordinate_systems('Instrument','Ship', inverse)
                
    def create_transformation_matrix(self, attitude, beamconfig):
        try:
            return self.R
        except AttributeError:
            R = RotationMatrix()
            hdg, pitch, roll = self.attitude
            if beamconfig.facing == 'Up': # if upward looking, rotate 180 over S-axis 
                roll+=np.pi
            if self.inverse:
                self.R = R(hdg, pitch, roll).T
            else:
                self.R = R(hdg, pitch, roll)
            return self.R

class TransformSFU_XYZ(TransformXYZ_SFU):
    ''' Transformation of SFU to XYZ using the angles set to transform from XYZ to SFU '''
    def __init__(self, hdg, pitch, roll, inverse = False):
        super().__init__(hdg, pitch, roll, not inverse)


class TransformENU_SFU(TransformSFU_ENU):
    ''' Transformation of ENU to SFU using the angles set to transform from SFU to ENU '''
    def __init__(self, inverse = False):
        super().__init__(not inverse)

        
class TransformENU_XYZ(TransformXYZ_ENU):
    ''' Transformation of ENU to XYZ using the angles set to transform from XYZ to ENU '''
    def __init__(self, inverse = False):
        super().__init__(not inverse)
        

class TransformXYZ_BEAM(TransformBEAM_XYZ):
    ''' Transformation of XYZ to BEAM using the values set to transform from BEAM to XYZ'''
    def __init__(self, inverse = False):
        super().__init__(not inverse)
    

        
class Altitude(Coroutine):
    '''Altitude

    A class to convert range measurements to distance to the sea bed
    (altitude).

    Parameters
    ----------
    mount_hdg : float (0.0)
        angle at which the instrument is mounted, as offset for heading
        angle (rad)
    mount_pitch : float (0.0)
        angle at which this instrument is mounted, as offset for the
        pitch angle (rad)
    mount roll : float (0.0)
        angle at which the instrument is mounted, as offset for the roll
        angle (rad)

    The angles are given in radians.

    Notes
    -----
    
    For the glider, the mount angle for heading is 0, for pitch 11
    degrees, and for the roll it depends on how the glider is
    assembled. This offset angle can be found if
    i) bottom track is available
    ii) the sea bed can be assumed reasonably flat

    Then, for the correct roll offset, the four ranges should collapse
    onto one curve.

    '''
    PARAMS = dict(bottom_track = ['Range'])
    CACHE = {}
    
    def __init__(self, mount_hdg=0, mount_pitch=0, mount_roll=0):
        super().__init__()
        self.Rxyz_fsu = RotationMatrix()(mount_hdg, mount_pitch, mount_roll)
        self.Rfsu_enu = RotationMatrix()
        self.coro_fun = self.coro_transform_ensembles()
        
    @coroutine
    def coro_transform_ensembles(self):
        ''' coroutine  transforming ensembles.'''
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                self.transform_range_in_ensemble(ens)
                self.send(ens)
        self.close_coroutine()

    def transform_range_in_ensemble(self, ens):
        '''Transforms the bottom_track_range fields in each ensemble.

        Parameters
        ----------
        ens : :class:`rdi_reader.Ensemble`
            Ensemble dictionary

        '''
        beam_vectors = self.get_beam_vectors(ens)
        hdg = ens['variable_leader']['Heading']*np.pi/180
        pitch = ens['variable_leader']['Pitch']*np.pi/180
        roll = ens['variable_leader']['Roll']*np.pi/180.
        R = self.Rfsu_enu(hdg, pitch, roll) @ self.Rxyz_fsu
        x,y,z,_ = R @ beam_vectors
        bt = ens['bottom_track']
        for i, (_x, _y, _z) in enumerate(zip(x,y,z)):
            bt['Range%d'%(i+1)] = _z # We want positive ranges.
        
    def get_beam_vectors(self, ens):
        '''Get the measured ranges as vectors
        
        Parameters
        ----------
        ens : :class:`rdi_reader.Ensemble`
            Ensemble dictionary
        
        Returns
        -------
        np.array (4,4) 
            the range measurements as vectors represented in the
            device's coordinate system (xyz)
        
        Notes
        -----
        It seems that the range measurements are expressed as their projected values
        onto the system's z coordinate. In order to obtain the correct ranges along
        the acoustic beams, the scaling has to be reversed (by the division of 
        cos(theta)).
        '''
        try:
            vectors, n_beams = self.CACHE['vectors']
            
        except KeyError:
            theta, beam_pattern, facing, n_beams = self.get_beam_configuration(ens)
            c = int(beam_pattern=='Convex')*2-1
            sn = c * np.sin(theta)
            cs = np.cos(theta)
            vectors = np.array([[sn,   0, cs, 0],
                                [ -sn,   0, cs, 0],
                                [  0,  -sn, cs, 0],
                                [  0, sn, cs, 0]]).T
            vectors /= np.cos(theta) # compensates for the mapping on the instruments z-axis
            self.CACHE['vectors'] = vectors, n_beams
        B = np.diag([ens['bottom_track']['Range%d'%(i+1)] for i in range(n_beams)])
        beam_vectors = vectors @ B
        return beam_vectors
    
            
    def get_beam_configuration(self, ens):
        ''' Gets the configuration of the beam settings.
        
        Parameters
        ----------
        ens : :class:`rdi_reader.Ensemble`
            Ensemble dictionary
        
        Returns
        -------
        tuple of (float, str, str, int)
            floatbeam angle theta (rad), 
            beam_pattern (convex/concave), 
            facing (up/down),
            number of beams
        '''
        fixed_leader = ens['fixed_leader']
        beam_angle, _ = fixed_leader['Beam_Angle'].split()
        beam_pattern = fixed_leader['Beam_Pattern']
        facing = fixed_leader['Xdcr_Facing']
        n_beams = fixed_leader['N_Beams']
        theta = float(beam_angle)*np.pi/180.
        return theta, beam_pattern, facing, n_beams
