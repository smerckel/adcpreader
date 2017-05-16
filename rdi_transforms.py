import numpy as np

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


class Transform(object):
    ''' Base transform class.

    Implements the transformations and multiplication, but not the definition of the rotation matrix.

    This class should be subclassed.
    '''
    # which parameters should be transformed
    PARAMS = dict(velocity = ['Velocity'],
                  bottom_track = ['BTVel'])

    def __init__(self, inverse = False):
        self.inverse = inverse

    def __mul__(self, ri):
        ''' Creates a create_rotation_matrix() method from the left and right arguments of the * operator. '''
        T = Transform()
        T.create_rotation_matrix = lambda *x: self.create_rotation_matrix(*x)*ri.create_rotation_matrix(*x)
        return T

        
    def transform_velocities_in_ensemble(self, ens):
        ''' Transforms the velocities in given ensemble. Depending on the rotation matrix, the
            heading/pitch/roll information may or may not be used.
        '''
        alpha = ens['variable_leader']['Heading']*np.pi/180.
        beta = ens['variable_leader']['Pitch']*np.pi/180.
        gamma = ens['variable_leader']['Roll']*np.pi/180.
        R = self.create_rotation_matrix(alpha, beta, gamma)
        self.__transform_velocities_in_ensemble(ens, R)
        
    def __transform_velocities_in_ensemble(self, ens, R):
        for k, v in self.PARAMS.items():
            for _v in v:
                x = np.matrix([ens[k]['%s%d'%(_v,i+1)] for i in range(4)])
                if x.shape[0] == 1: # for bottom track values
                    xp = np.array(R * x.T)
                    for i in range(4):
                        ens[k]['%s%d'%(_v, i+1)] = float(xp[i])
                else:
                    xp = np.array(R * x)
                    for i in range(4):
                        ens[k]['%s%d'%(_v, i+1)] = xp[i]
        
    def gen(self, ensembles):
        ''' generator yielding transformed ensembles.'''
        for ens in ensembles:
            self.transform_velocities_in_ensemble(ens)
            yield ens
            
class TransformFSU_ENU(Transform):
    def __init__(self, inverse = False):
        super().__init__(inverse)
        self.static = False
    
    def create_rotation_matrix(self, alpha, beta, gamma):
        R = RotationMatrix()
        if self.inverse:
            return R(alpha, beta, gamma).T
        else:
            return R(alpha, beta, gamma)

class TransformXYZ_FSU(Transform):
    def __init__(self, alpha, beta, gamma, inverse = False):
        super().__init__(inverse)
        self.static = True
        R = RotationMatrix()
        if self.inverse:
            self.R = R(alpha, beta, gamma).T
        else:
            self.R = R(alpha, beta, gamma)
    def create_rotation_matrix(self, *p):
        return self.R

class TransformFSU_XYZ(TransformXYZ_FSU):
    ''' Transformation of FSU to XYZ using the angles set to transform from XYZ to FSU '''
    def __init__(self, alpha, beta, gamma, inverse = False):
        super().__init__(alpha, beta, gamma, not inverse)

class TransformENU_FSU(TransformFSU_ENU):
    ''' Transformation of FSU to XYZ using the angles set to transform from XYZ to FSU '''
    def __init__(self, inverse = False):
        super().__init__(not inverse)
        
    

if __name__ == "__main__":
    filename = "PF230519.PD0"

    import pd0
    import glob

    filenames = glob.glob("/home/lucas/gliderdata/subex2016/adcp/PF*.PD0")
    filenames.sort()

    bindata = pd0.PD0()
    ensembles = bindata.ensemble_generator(filenames)


    #t1 = TransformFSU_ENU(inverse=True)
    #t2 = TransformXYZ_FSU(alpha=0, beta=0.1919, gamma=0, inverse=True)
    
    t1 = TransformENU_FSU()
    t2 = TransformFSU_XYZ(alpha=0, beta=0.1919, gamma=0)
    t3 = TransformXYZ_FSU(alpha=0, beta=0.2239, gamma=0.05)

    t4 = t3*t2*t1

    ensembles = t4.gen(ensembles)

    _v = []
    for i,ens in enumerate(ensembles):
        _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
        if i==2000:
            break

    vx,vy,vz = np.array(_v).T
