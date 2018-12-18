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

from rdi import __VERSION__
from rdi.coroutine import coroutine, Coroutine

Attitude = namedtuple('Attitude', 'hdg pitch roll')
Beamconfig = namedtuple('Beamconfig', 'a b c d facing')


class RotationMatrix(object):
    ''' A rotation matrix class as defined in the RDI manual '''
    def create_matrix(self, heading, pitch, roll):
        CH = np.cos(heading) 
        CP = np.cos(pitch)   
        CR = np.cos(roll)    
        SH = np.sin(heading) 
        SP = np.sin(pitch)   
        SR = np.sin(roll)    
        M = np.matrix([[CH*CR+SH*SP*SR, SH*CP, CH*SR-SH*SP*CR, 0],
                       [-SH*CR+CH*SP*SR, CH*CP,-SH*SR-CH*SP*CR,0],
                       [-CP*SR, SP, CP*CR, 0],
                       [0, 0 ,0, 1]])
        return M
        
    def __call__(self, heading, pitch, roll):
        return self.create_matrix(heading, pitch, roll)


    
class TransformMatrix(object):
    def create_matrix(self, a, b, c, d):
        M = np.matrix([[c*a, -c*a, 0, 0],
                       [0  ,    0, -c*a, c*a],
                       [b  ,    b,    b,   b],
                       [d  ,    d,   -d,  -d]])
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
    # If parameters are modified, propagate the modifications back into the ens data (pitch for example).
    UPDATE_CORRECTIONS = True
    
    def __init__(self, inverse = False):
        super().__init__()
        self.inverse = inverse
        self.hooks = {}
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
        
    def __mul__(self, ri):
        ''' Creates a create_transformation_matrix() method from the left and right arguments of the * operator. '''
        T = Transform()
        T.create_transformation_matrix = lambda *x: self.create_transformation_matrix(*x) * ri.create_transformation_matrix(*x)
        if self.transformed_coordinate_system:
            T.transformed_coordinate_system = self.transformed_coordinate_system
        else:
            T.transformed_coordinate_system = ri.transformed_coordinate_system
        T.hooks = dict((k,v) for k,v in chain(ri.hooks.items(), self.hooks.items()))
        return T

    def attitude_correction(self, hdg, ptch, roll):
        ''' Corrects the heading, pitch and roll using a callable with key "attitude_correction" in self.hooks'''
        try:
            f = self.hooks['attitude_correction']
        except KeyError:
            # no function found, return as is.
            return hdg, ptch, roll
        else:
            return f(hdg, ptch, roll)

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
        hdg = ens['variable_leader']['Heading']*np.pi/180.
        pitch = ens['variable_leader']['Pitch']*np.pi/180.
        roll = ens['variable_leader']['Roll']*np.pi/180.
        hdg, pitch, roll = self.attitude_correction(hdg, pitch, roll)
        if self.UPDATE_CORRECTIONS:
            ens['variable_leader']['Heading'] = hdg/np.pi*180.
            ens['variable_leader']['Pitch'] = pitch/np.pi*180.
            ens['variable_leader']['Roll'] = roll/np.pi*180.

        attitude = Attitude(hdg, pitch, roll)

        a,b,c,d, facing = self.get_beam_configuration(ens)
        beamconfig = Beamconfig(a, b,c,d, facing)

        R = self.create_transformation_matrix(attitude, beamconfig)
        params = self.get_params(ens)
        self.__transform_velocities_in_ensemble(ens, R, params)
        self.update_coordinate_frame_setting(ens)
        
    def __transform_velocities_in_ensemble(self, ens, R, params):
        for k, v in params.items():

            for _v in v:
                # make a note of the mask of this variable, if any.
                try:
                    mask = ens[k]['%s%d'%(_v,1)].mask
                except AttributeError:
                    mask = None
                x = np.matrix([ens[k]['%s%d'%(_v,i+1)] for i in range(4)])
                if x.shape[0] == 1: # for bottom track values
                    xp = np.array(R * x.T)
                    for i in range(4):
                        ens[k]['%s%d'%(_v, i+1)] = float(xp[i])
                else:
                    xp = np.array(R * x)
                    for i in range(4):
                        if mask is None: 
                            ens[k]['%s%d'%(_v, i+1)] = xp[i]                            
                        else: #apply the mask again
                            ens[k]['%s%d'%(_v, i+1)] = np.ma.masked_array(xp[i], mask)
                            
    def update_coordinate_frame_setting(self, ens):
        ''' Writes the new coordinate frame setting and records the original setting. '''
        if self.transformed_coordinate_system:
            ens['fixed_leader']['OriginalCoordXfrm'] = ens['fixed_leader']['CoordXfrm']
            ens['fixed_leader']['CoordXfrm'] = self.transformed_coordinate_system
        else:
            raise ValueError('Transformed_coordinate_system is NOT set!')
        
            
class TransformSFU_ENU(Transform):
    def __init__(self, inverse = False):
        super().__init__(inverse)
        if inverse:
            self.transformed_coordinate_system = 'Ship'
        else:
            self.transformed_coordinate_system = 'Earth'

    def create_transformation_matrix(self, attitude, beamconfig):
        R = RotationMatrix()
        if self.inverse:
            return R(attitude.hdg, attitude.pitch, attitude.roll).T
        else:
            return R(attitude.hdg, attitude.pitch, attitude.roll)

class TransformXYZ_ENU(Transform):
    def __init__(self, inverse = False):
        super().__init__(inverse)
        if inverse:
            self.transformed_coordinate_system = 'Instrument'
        else:
            self.transformed_coordinate_system = 'Earth'

    def create_transformation_matrix(self, attitude, beamconfig):
        R = RotationMatrix()
        if self.inverse:
            return R(attitude.hdg, attitude.pitch, attitude.roll).T
        else:
            return R(attitude.hdg, attitude.pitch, attitude.roll)


class TransformRotation(Transform):
    def __init__(self, hdg, pitch, roll, transformed_coordinate_system=None):
        inverse = False
        super().__init__(inverse)
        R = RotationMatrix()
        self.R = R(hdg, pitch, roll)
        self.transformed_coordinate_system = transformed_coordinate_system
            
    def create_transformation_matrix(self, *p):
        return self.R

class TransformBEAM_XYZ(Transform):
    def __init__(self,inverse = False):
        super().__init__(inverse)
        if self.inverse:
            self.transformed_coordinate_system = 'Beam'
        else:
            self.transformed_coordinate_system = 'Instrument'
        
    def create_transformation_matrix(self, attitude, beamconfig):
        try:
            return self.R
        except AttributeError:
            R = TransformMatrix()
            self.R = R(beamconfig.a, beamconfig.b, beamconfig.c, beamconfig.d)
            if self.inverse:
                self.R = np.linalg.inv(self.R) # Transpose is not okay for non-rotational matrices.
            return self.R
        
class TransformXYZ_SFU(Transform):
    def __init__(self, hdg, pitch, roll, inverse = False):
        super().__init__(inverse)
        self.attitude = Attitude(hdg, pitch, roll)
        if self.inverse:
            self.transformed_coordinate_system = 'Instrument'
        else:
            self.transformed_coordinate_system = 'Ship'
                
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
    
    

