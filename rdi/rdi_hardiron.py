import numpy as np


class HardIron(object):

    def __init__(self, Hvector, declination, inclination):
        '''
        :param Hvector: Hard iron vector
        :param declination: local angle of declination (difference between magnetic and true north)
        :param inclination: local angle of inclination (angle at which magnetic vector points into the earth

        :type Hvector: list/array of size 3
        :type declination: float
        :type inclination: float
        '''
        self.Hvector = Hvector
        self.declination = declination
        self.inclination = inclination

    def attitude_errors(self, pitch, hdg):
        '''
        Computes errors in pitch and heading, for measured pitch and heading.

        :param pitch: measured pitch
        :param hdg: measured heading
        :type pitch: float or array
        :type hdg: float or array
        :return delta: error in heading
        :return epsilon: error in pitch
        :rtype delta: float or array
        :rtype epsilon: float or array
        '''
        # adapted from dvl_pitch_correction
        #declination and inclination from https://ngdc.noaa.gov/geomag-web/#igrfwmm
        phi=np.pi/180*self.inclination
        a=np.pi/2-hdg
        b=pitch
        Hx, Hy, Hz = self.Hvector
        
        delta = -((Hx*np.sin(a)-Hy*np.cos(a))*np.sin(phi)+((Hy*np.sin(a)+Hx*np.cos(a))*np.sin(b)+Hz*np.cos(b))*np.cos(phi))/np.sin(b)
        epsilon = (((Hy*np.sin(a)+Hx*np.cos(a))*np.cos(b)*np.sin(b)+Hz*np.cos(b)**2)*np.sin(phi)**2+(Hy*np.cos(a)-Hx*np.sin(a))*np.cos(b)*np.cos(phi)*np.sin(phi)-Hz)/(np.sin(b)*np.sin(phi))
        # delta = ((Hx*np.sin(a)-Hy*np.cos(a))*np.sin(phi)+
        #          ((-Hy*np.sin(a)-Hx*np.cos(a))*np.sin(b)-Hz*np.cos(b))*np.cos(phi))/np.sin(b)
        # epsilon = -(((Hy*np.sin(a)+Hx*np.cos(a))*np.cos(b)*np.sin(b)+
        #              Hz*np.cos(b)**2)*np.sin(phi)**2+
        #             (Hx*np.sin(a)-Hy*np.cos(a))*np.cos(b)*np.cos(phi)*np.sin(phi)-Hz)/(np.sin(b)*np.sin(phi))
        return delta, epsilon

    def attitude_correction(self, hdg, pitch, roll):
        ''' Apply attitude error correction

            :param hdg: measured heading
            :param pitch: measured pitch
            :param roll: measured roll
            :type hdg: float or array
            :type pitch: float or array
            :type roll: float or array
            :return m_hdg: modified heading
            :return m_pitch: modified pitch
            :return m_roll: modified roll
        '''
        if isinstance(pitch, float):
            if pitch==0:
                delta, epsilon = 0., 0.
            else:
                delta, epsilon = self.attitude_errors(pitch, hdg)
        else:
            if np.any(pitch==0):
                delta = np.zeros_like(pitch)
                epsilon = np.zeros_like(pitch)
                j = np.where(pitch!=0)[0]
                delta[j], epsilon[j] = self.attitude_errors(pitch[j], hdg[j])
            else:
                delta, epsilon = self.attitude_errors(pitch, hdg)
        return hdg-delta-self.declination*np.pi/180, pitch-epsilon, roll
