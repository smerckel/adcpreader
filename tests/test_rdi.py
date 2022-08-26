import arrow
from hashlib import md5
import os
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


class TestWriter:
    
    def remove_file(self, fn):
        os.unlink(fn)

    def checksum_file(self, fn):
        with open(fn, 'rb') as fp:
            M = md5(fp.read())
        return M.hexdigest()

    def test_ascii(self):
        tmpfile = 'output.asc'
        source = rdi.rdi_reader.PD0()
            
        with open(tmpfile, 'w') as fp:
            data = rdi.rdi_writer.AsciiWriter(output_file=fp)

            pipeline = source | data
            pipeline.process([pd0_filename])

        checksum = self.checksum_file(tmpfile)
        self.remove_file(tmpfile)
        assert checksum == 'f0777f66ce6c01c47b23b447cf8fb1f3'

    def test_nc(self):
        tmpfile = 'output.nc'
        source = rdi.rdi_reader.PD0()
            
        data = rdi.rdi_writer.NetCDFWriter(tmpfile)
        
        pipeline = source | data

        with data: 
            pipeline.process([pd0_filename])

        checksum = self.checksum_file(tmpfile)
        self.remove_file(tmpfile)
        assert checksum == '4ce1a6e9911f5e3ba6fd1f41dfc19e21'

        
if __name__ == "__main__":

    tq = TestWriter()
    data = tq.test_nc()
    
        
