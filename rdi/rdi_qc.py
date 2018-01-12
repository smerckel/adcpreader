import numpy as np

from rdi import __VERSION__

# default setting for if true, all ensembles that have all data
# blanked out because of some quality check will silently be dropped.
DROP_MASKED_ENSEMBLES_BY_DEFAULT = False
    

class QualityControl(object):
    ''' Quality Control base class

        Implements conditions to make arrays masked arrays,
    
        scalars that don't pass the condition are set to nan.
    '''
    def __init__(self, drop_masked_ensembles=None):
        self.conditions = list()
        self.operations = {">":self.discard_greater,
                           ">=":self.discard_greater_equal,
                           "<":self.discard_less,
                           "<=":self.discard_less_equal,
                           "||>":self.discard_abs_greater,
                           "||>=":self.discard_abs_greater_equal,
                           "||<":self.discard_abs_less,
                           "||<=":self.discard_abs_less_equal}
        if drop_masked_ensembles is None:
            self.drop_masked_ensembles = DROP_MASKED_ENSEMBLES_BY_DEFAULT
        else:
            self.drop_masked_ensembles = drop_masked_ensembles
            
    def __call__(self, ensembles):
        ''' returns the ensemble generator '''
        return self.gen(ensembles)

    def gen(self, ensembles):
        ''' generator, returning checked ensembles '''
        for i, ens in enumerate(ensembles):
            keep_ensemble = self.check_ensemble(ens)
            if keep_ensemble or not self.drop_masked_ensembles:
                yield ens
            else:
                continue # ensemble is dropped

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
    VECTORS = 'velocity correlation echo percent_good'.split()
    SCALARS = ['bottom_track']
    
    def __init__(self, drop_masked_ensembles=None):
        super().__init__(drop_masked_ensembles)
        
    def set_discard_condition(self, section, parameter, operator, value):
        ''' Set a condition to discard readings.

        section: section name of the data block. Example: velocity
        parameter: name of parameter in this section. Example Velocity1
        operator: comparison operator. Example: ">" or "||>"
        value:    the value to compare with.
        '''
        self.conditions.append((section, parameter, operator, value))

    def check_ensemble(self, ens):
        keep_ensemble = True
        mask_ensemble = False
        
        for section, parameter, operator, value in self.conditions:
            if section != 'variable_leader':
                continue
            v = ens[section][parameter]
            f = self.operations[operator]
            _v = f(v, value)
            # we don't put nans in the variable leader. If the check causes a positive, mask the
            # the variables in the sections SCALARS and VECTORS (see above).
            # ens[section][parameter] = _v
            if np.isnan(_v):
                mask_ensemble = True
                keep_ensemble = False
        if mask_ensemble:
            for section in ValueLimit.VECTORS:
                if section not in ens.keys():
                    continue
                for k, v in ens[section].items():
                    ens[section][k]=np.ma.masked_array(v, True)
            for section in ValueLimit.SCALARS:
                if section not in ens.keys():
                    continue
                for k, v in ens[section].items():
                    ens[section][k]=np.nan
        else:
            for section, parameter, operator, value in self.conditions:
                if section == 'variable_leader':
                    continue # already done
                if section not in ens.keys():
                    continue
                v = ens[section][parameter]
                f = self.operations[operator]
                _v = f(v, value)
                ens[section][parameter] = _v
                if np.isscalar(_v): # if parameter is scalar and nan, drop the ens.
                    if np.isnan(_v):
                        keep_ensemble = False
                else:
                    if np.all(_v.mask): # if all values are masked, drop it too.
                        keep_ensemble = False
        return keep_ensemble
            


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
        return True # always return the ensemble

class Counter(object):
    ''' An ensemble counter class.

    This class merely counts the number of ensembles that pass through the pipeline at this stage.
    This implies that no ensemble is modified.
    
    The number of ensembles counted are stored in the property counts.

    An instance of this class can be placed at more than one position within the pipeline. The counts 
    property is a list that reflects the positions where the counter is placed.

    '''
    
    def __init__(self):
        self.counts = []

    def __call__(self, ensembles):
        return self.gen(ensembles)
    
    def gen(self, ensembles):
        for i, ens in enumerate(ensembles):
            yield ens
        self.counts.append(i)
