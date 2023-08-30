import re

import numpy as np

from adcpreader import __VERSION__

from adcpreader.coroutine import coroutine, Coroutine


# default setting for if true, all ensembles that have all data
# blanked out because of some quality check will silently be dropped.
DROP_MASKED_ENSEMBLES_BY_DEFAULT = False

            
    

class QualityControl(Coroutine):
    ''' Quality Control base class

        Implements conditions to make arrays masked arrays,
    
        scalars that don't pass the condition are set to nan.
    '''
    def __init__(self, drop_masked_ensembles=None):
        super().__init__()
        self.rules = dict(default=list(),
                          regex=list(),
                          vl_default=list())

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
        self.coro_fun = self.coro_check_ensembles()
        
    @coroutine
    def coro_check_ensembles(self):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                keep_ensemble = self.check_ensemble(ens)
                if keep_ensemble or not self.drop_masked_ensembles:
                    self.send(ens)
                else:
                    continue # ensemble is dropped
        self.close_coroutine()


        
    def check_ensemble(self, ens):
        ''' check an ensemble. Should be subclassed.'''
        raise NotImplementedError

    def discard_greater(self, v, value):
        ''' discard values v that are greater than value '''
        condition = v>value
        return condition

    def discard_greater_equal(self, v, value):
        ''' discard values v that are greater or equal than value '''
        condition = v>=value
        return condition

    def discard_less(self, v, value):
        ''' discard values v that are less than value '''
        condition = v<value
        return condition

    def discard_less_equal(self, v, value):
        ''' discard values v that are less or equal than value '''
        condition = v<value
        return condition

    def discard_abs_greater(self, v, value):
        ''' discard values v that are absolute greater than value '''
        condition = np.abs(v)>value
        return condition

    def discard_abs_greater_equal(self, v, value):
        ''' discard values v that are absolute greater or equal than value '''
        condition = np.abs(v)>=value
        return condition

    def discard_abs_less(self, v, value):
        ''' discard values v that are absolute less than value '''
        condition = np.abs(v)<value
        return condition

    def discard_abs_less_equal(self, v, value):
        ''' discard values v that are absolute less or equal than value '''
        condition = np.abs(v)<value
        return condition

    def apply_condition(self, condition, v):
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

    def mask_parameter_regex(self, section, parameter_regex, operator, value, boolean='OR'):
        regex = re.compile(parameter_regex)
        if section=='variable_leader':
            raise ValueError("not tested, fix me")
            self.rules['vl_regex'].append( (section, regex, operator, value) )
        else:
            self.rules['regex'].append( (section, regex, operator, value, boolean) )

        
    def mask_parameter(self, section, parameter, operator, value, dependent_parameters=dict()):
        ''' Set a condition to discard readings.

        Parameters
        ----------
        section : string
            section name of the data block. Example: velocity
        parameter : string
            name of parameter in this section. Example Velocity1
        operator : string
            comparison operator. Example: ">" or "||>"
        value : float
            the value to compare with.
        dependent_parameters : dict
            If not empty, the mask condition is also applied to the "dependent parameters".
        
        Example
        -------
        mask_parameter("velocity", "Velocity1", "||>", 0.1, 
                        dependent_parameters=dict(bottom_track=["BTVel1", "BTVel2"]))

        Table of implemented operators:
        
        +--------+------------------------------+
        |   >    | greater than                 |
        +--------+------------------------------+
        |   >=   | greater equal than           |
        +--------+------------------------------+
        |   <    | smaller than                 |
        +--------+------------------------------+
        |   <=   | smaller equal than           |
        +--------+------------------------------+
        |  ||>   | absolute greater than        |
        +--------+------------------------------+
        |  ||>=  | absolute greater equal than  |
        +--------+------------------------------+
        |  ||<   | absolute smaller than        |
        +--------+------------------------------+
        |  ||<=  | absolute smaller equal than  |
        +--------+------------------------------+
        
        '''
        if section == 'variable_leader':
            self.rules['vl_default'].append((section, parameter, operator, value,
                                             dependent_parameters))
        else:
            self.rules['default'].append((section, parameter, operator, value,
                                          dependent_parameters))

    def check_ensemble(self, ens):
        keep_ensemble = True
        mask_ensemble = False
        # process the variable leader first, to see if we need to mask the ensemble
        for section, parameter, operator, value, dependent_parameters in self.rules['vl_default']:
            if dependent_parameters:
                raise NotImplementedError('It is not possible (yet) to apply a mask to other variables, for the variable leader')
            v = ens[section][parameter]
            f = self.operations[operator]
            condition = f(v, value)
            # we don't put nans in the variable leader. If the check causes a positive, mask the
            # the variables in the sections SCALARS and VECTORS (see above).
            # ens[section][parameter] = _v
            if np.isnan(condition):
                mask_ensemble = True
                keep_ensemble = False
        # if the variable_leader is such that the enemble is to be masked, do it.
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
            # variable leader does not require the ensemble to be
            # masked. See if there is any particular parameter to be
            # masked.
            for section, parameter, operator, value, dependent_parameters in self.rules['default']:
                if section not in ens.keys():
                    continue
                v = ens[section][parameter]
                f = self.operations[operator]
                condition = f(v, value)
                ens[section][parameter] = self.apply_condition(condition, v)
                # now apply the condition to other parameters if required:
                for s, ps in dependent_parameters.items():
                    for p in ps:
                        ens[s][p] = self.apply_condition(condition, ens[s][p])
                try:
                    keep_ensemble = not condition
                except ValueError as e:
                    if e.args[0] == 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()':
                        keep_ensemble = not np.all(condition)
                    else:
                        raise e # unexpected error, reraise just in case.
            # do things slightly different if regex is used
            for section, parameter, operator, value, boolean in self.rules['regex']:
                if section not in ens.keys():
                    continue
                matching_parameters = [i for i in ens[section].keys() if parameter.match(i)]
                f = self.operations[operator]
                for i, p in enumerate(matching_parameters):
                    v = ens[section][p]
                    if i==0:
                        condition = f(v, value)
                    else:
                        if boolean == 'OR':
                            condition |= f(v, value)
                        elif boolean == 'AND':
                            condition &= f(v, value)
                        else:
                            raise NotImplementedError('Boolean type not implemented')
                if matching_parameters[0].startswith("Velocity"):
                    try:
                        print(self.ID, ens['variable_leader']['Ensnum'])
                        print(condition)
                        print()
                        input("br")
                    except:
                        pass
                for p in matching_parameters:
                    ens[section][p] = self.apply_condition(condition, ens[section][p])
                try:
                    keep_ensemble = not condition
                except ValueError as e:
                    if e.args[0] == 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()':
                        keep_ensemble = not np.all(condition)
                    else:
                        raise e # unexpected error, reraise just in case.
        return keep_ensemble
            


class SNRLimit(QualityControl):
    '''Signal to Noise Ratio limit

    Masks or drops ensembles for which the SNR fails to exceed the threshold

    Parameters
    ----------
    SNR_limit: float
        SNR threshold (default 10)
    noise_floor_db: float
        noise floor in dB (default 26.1)


    
    The SNR is calculated according to
    
    .. math::
    
         SNR = 10^{(E-E0)/10}

    where :math:`E` is the echo intensity in dB, and :math:`E0` the noise floor.

    '''
    
    def __init__(self, SNR_limit = 10, noise_floor_db = 26.1):
        super().__init__()
        self.SNR_limit = SNR_limit
        self.noise_floor_db = noise_floor_db

    def SNR(self, echointensity):
        return 10**((echointensity-self.noise_floor_db)/10)
    
    def check_ensemble(self, ens):
        ''' '''
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
            ens['velocity'][s] = self.apply_condition(condition, ens['velocity'][s])
            s="SNR%d"%(i+1)
            ens['echo'][s] = SNR[i]
        return True # always return the ensemble


class AcousticAmplitudeLimit(QualityControl):

    ''' Acoustic Amplitude Limit

    Masks bins where the amplitude less than a given threshold.

    Parameters
    ----------
    amplitude_limit: float
        minimum required amplitude. (default 75)

    .. note::

        This limiter never drops an ensemble.
    '''
    def __init__(self, amplitude_limit = 75):
        super().__init__()
        self.amplitude_limit = amplitude_limit

    def check_ensemble(self, ens):
        ''' '''
        nbeams = ens['fixed_leader']['N_Beams']
        s = ["Echo%d"%(i+1) for i in range(nbeams)]
        amplitudes = [ens['echo'][_s]  for _s in s]
        for i,amplitude in enumerate(amplitudes):
            if i:
                condition|= amplitude > self.amplitude_limit
            else:
                condition = amplitude > self.amplitude_limit
        for i in range(nbeams):
            s="Velocity%d"%(i+1)
            ens['velocity'][s] = self.apply_condition(condition, ens['velocity'][s])
        return True # always return the ensemble

class MaskBins(QualityControl):
    ''' Mask bins

    This operation masks all specified bins.

    Parameters
    ----------
    masked_bins: list of integers
         list of bin numbers that are to be masked. Starts with 0 for the first bin. (default [])
    '''
    def __init__(self, masked_bins = []):
        super().__init__()
        self.masked_bins = masked_bins
        
    def check_ensemble(self, ens):
        ''' '''
        nbeams = ens['fixed_leader']['N_Beams']
        mask = np.zeros_like(ens['velocity']['Velocity1'], dtype=bool)
        for i in self.masked_bins:
            mask[i]=True
        for i in range(nbeams):
            s="Velocity%d"%(i+1)
            ens['velocity'][s] = self.apply_condition(mask, ens['velocity'][s])
        return True # always return the ensemble
        

        n_beams = ens['fixed_leader']['NBeams']
        vel_dict = ens['velocity']
        
class Counter(Coroutine):
    ''' An ensemble counter class.

    This class merely counts the number of ensembles that pass through the pipeline at this stage.
    This implies that no ensemble is modified.

    Parameters
    ----------
    verbose: bool
        Sets whether the sequential number of the processed ensemble is to be printed. (default False)

    
    The number of ensembles counted are stored in the property counts.

    An instance of this class can be placed at more than one position within the pipeline. The counts 
    property is a list that reflects the positions where the counter is placed.

    '''
    
    def __init__(self, verbose=False):
        super().__init__()
        self.counts = 0
        self.coro_fun = self.coro_counter(verbose)

    @coroutine
    def coro_counter(self, verbose=False):
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                n = ens['variable_leader']['Ensnum']
                self.counts+=1
                if verbose:
                    print("Ensemble : {:4d}/{:5d}".format(n, self.counts))
                self.send(ens)
        self.close_coroutine()


