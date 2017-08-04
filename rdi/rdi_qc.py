import numpy as np

from rdi import __VERSION__


class QualityControl(object):
    ''' Quality Control base class

        Implements conditions to make arrays masked arrays,
    
        scalars that don't pass the condition are set to nan.
    '''
    
    def __init__(self):
        self.conditions = list()
        self.operations = {">":self.discard_greater,
                           ">=":self.discard_greater_equal,
                           "<":self.discard_less,
                           "<=":self.discard_less_equal,
                           "||>":self.discard_abs_greater,
                           "||>=":self.discard_abs_greater_equal,
                           "||<":self.discard_abs_less,
                           "||<=":self.discard_abs_less_equal}
        
    def __call__(self, ensembles):
        ''' returns the ensemble generator '''
        return self.gen(ensembles)

    def gen(self, ensembles):
        ''' generator, returning checked ensembles '''
        for ens in ensembles:
            self.check_ensemble(ens)
            yield ens

    def check_ensemble(self, ens):
        ''' check an ensemble. Should be subclassed.'''
        raise NotImplementedError

    def discard_greater(self, v, value):
        ''' discard values v that are greater than value '''
        condition = v>value
        return self.apply_condition(v, condition)

    def discard_greater_equal(self, v, value):
        ''' discard values v that are greater or equal than value '''
        condition = v>=value
        return self.apply_condition(v, condition)

    def discard_less(self, v, value):
        ''' discard values v that are less than value '''
        condition = v<value
        return self.apply_condition(v, condition)

    def discard_less_equal(self, v, value):
        ''' discard values v that are less or equal than value '''
        condition = v<value
        return self.apply_condition(v, condition)

    def discard_abs_greater(self, v, value):
        ''' discard values v that are absolute greater than value '''
        condition = np.abs(v)>value
        return self.apply_condition(v, condition)

    def discard_abs_greater_equal(self, v, value):
        ''' discard values v that are absolute greater or equal than value '''
        condition = np.abs(v)>=value
        return self.apply_condition(v, condition)

    def discard_abs_less(self, v, value):
        ''' discard values v that are absolute less than value '''
        condition = np.abs(v)<value
        return self.apply_condition(v, condition)

    def discard_abs_less_equal(self, v, value):
        ''' discard values v that are absolute less or equal than value '''
        condition = np.abs(v)<value
        return self.apply_condition(v, condition)

    def apply_condition(self, v, condition):
        try:
            v.mask |= condition 
        except AttributeError as e:
            if e.args[0] == "'numpy.ndarray' object has no attribute 'mask'":
                v = np.ma.masked_array(v, condition)
            else:
                if condition:
                    v = np.nan
        return v
    
                
class ValueLimit(QualityControl):
    ''' Qualtiy Control class to mask values that are exceeding some limit.'''
    def __init__(self):
        super().__init__()
        
    def set_discard_condition(self, section, parameter, operator, value):
        ''' Set a condition to discard readings.

        section: section name of the data block. Example: velocity
        parameter: name of parameter in this section. Example Velocity1
        operator: comparison operator. Example: ">" or "||>"
        value:    the value to compare with.
        '''
        self.conditions.append((section, parameter, operator, value))

    def check_ensemble(self, ens):
        for section, parameter, operator, value in self.conditions:
            v = ens[section][parameter]
            f = self.operations[operator]
            _v = f(v, value)
            ens[section][parameter] = _v
            
            


class SNRLimit(QualityControl):
    def __init__(self, SNR_limit = 10, noise_floor_db = 26.1):
        super().__init__()
        self.SNR_limit = SNR_limit
        self.noise_floor_db = noise_floor_db

    def SNR(self, echointensity):
        return 10**((echointensity-self.noise_floor_db)/10)
    
    def check_ensemble(self, ens):
        nbeams = ens['fixed_leader']['N_Beams']
        s = ["Echo%d"%(i+1) for i in range(nbeams)]
        SNR = [self.SNR(ens['echo'][_s])  for _s in s]
        for i,snr in enumerate(SNR):
            if i:
                condition|= snr < self.SNR_limit
            else:
                condition = snr < self.SNR_limit
        for i in range(nbeams):
            s="Velocity%d"%(i+1)
            ens['velocity'][s] = self.apply_condition(ens['velocity'][s], condition)

