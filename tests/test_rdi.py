import arrow
import sys

import numpy as np

sys.path.insert(0, '..')
import rdi.rdi_reader, rdi.rdi_writer, rdi.rdi_qc

pd0_filename = '../data/PF230519.PD0'


class TestReader:

    def test_read_file(self):
        ''' Read a file and count the number of profiles.'''
        pd0 = rdi.rdi_reader.PD0()
        g = pd0.ensemble_generator_per_file(pd0_filename)
        for c, ens in enumerate(g):
            pass
        # This file holds 175 profiles, so c should be equal to 174
        assert c==174


    def test_process(self):
        ''' Build a simple pipeline using process '''

        source = rdi.rdi_reader.PD0()
        info = rdi.rdi_writer.Info("Header of PD0 data file", pause=False)
        data = rdi.rdi_writer.DataStructure()
        

        pipeline = source | info | data
        pipeline.process([pd0_filename])
        # Now data should have 175 time stamps
        assert data.Time.shape[0] == 175

    def test_process_timestamps(self):
        ''' Build a simple pipeline using process, and check whether we have correct time stamps.'''

        source = rdi.rdi_reader.PD0()
        info = rdi.rdi_writer.Info("Header of PD0 data file", pause=False)
        data = rdi.rdi_writer.DataStructure()
        

        pipeline = source | info | data
        pipeline.process([pd0_filename])
        t0 = arrow.get(data.Time[0]).ctime()
        assert t0 == 'Thu Jun 23 05:19:39 2016'


class TestQC:

    def test_mask_bins(self):
        source = rdi.rdi_reader.PD0()
        maskbins = rdi.rdi_qc.MaskBins(masked_bins=list(range(30)))
        data = rdi.rdi_writer.DataStructure()
        

        pipeline = source | maskbins | data
        pipeline.process([pd0_filename])
        assert data.velocity_east.mask.all() == True

    def test_value_limit(self):
        source = rdi.rdi_reader.PD0()
        value_limit = rdi.rdi_qc.ValueLimit()
        value_limit.mask_parameter("velocity", "Velocity1",  "||>", 0.1)
        data = rdi.rdi_writer.DataStructure()
        

        pipeline = source | value_limit | data
        pipeline.process([pd0_filename])
        assert data.velocity_east.max() <= 0.1 and data.velocity_north.max() > 0.1
        #return data

    def test_value_limit_dependent_parameters(self):
        source = rdi.rdi_reader.PD0()
        value_limit = rdi.rdi_qc.ValueLimit()
        dparams = dict(echo="Echo1 Echo2 Echo3 Echo4".split())
        value_limit.mask_parameter("velocity", "Velocity1",  "||>", -1, dparams)
        data = rdi.rdi_writer.DataStructure()
        pipeline = source | value_limit | data
        pipeline.process([pd0_filename])
        # All echo data should be masked as well.
        assert data.echo_Echo1.mask.all()
        
    def test_value_limit_regex(self):
        source = rdi.rdi_reader.PD0()
        value_limit = rdi.rdi_qc.ValueLimit()
        value_limit.mask_parameter_regex("velocity", "Velocity?",  "||>", 0.1)
        data = rdi.rdi_writer.DataStructure()
        

        pipeline = source | value_limit | data
        pipeline.process([pd0_filename])
        assert data.velocity_east.max() <= 0.1 and data.velocity_north.max() <= 0.1
        #return data
        
if __name__ == "__main__":

    tq = TestQC()
    data = tq.test_value_limit_dependent_parameters()
    
        
