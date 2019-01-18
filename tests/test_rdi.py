import unittest
import numpy as np

if 1: # use local install
    import sys
    sys.path.insert(0, '../rdi')
    import rdi_reader, rdi_writer, rdi_qc, rdi_transforms, rdi_corrections, rdi_backscatter
else:
    from rdi import rdi_reader, rdi_writer, rdi_qc, rdi_transforms, rdi_corrections, rdi_backscatter


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.filename = "../data/PF230519.PD0"

    def tearDown(self):
        pass

    def test_ascii(self):
        pd0 = rdi_reader.PD0()
        with open('PF230519.txt', 'w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            pd0.send_to(writer)
            pd0.process(self.filename)

    def test_ndf(self):
        pd0 = rdi_reader.PD0()
        writer = rdi_writer.NDFWriter('PF230519.ndf')
        pd0.send_to(writer)
        pd0.process(self.filename)

            
    def test_netcdf(self):
        pd0 = rdi_reader.PD0()
        writer = rdi_writer.NetCDFWriter('PF230519.nc')
        with writer:
            pd0.send_to(writer)
            pd0.process(self.filename)

    def test_qc(self):
        pd0 = rdi_reader.PD0()
        
        qc = rdi_qc.ValueLimit()
        qc.set_discard_condition('velocity', 'Velocity1', '||>', 1)
        with open('tmp','w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            pd0.send_to(qc)
            qc.send_to(writer)
            pd0.process(self.filename)
        
    def test_transforms_to_beams(self):
        pd0 = rdi_reader.PD0()

        t1 = rdi_transforms.TransformENU_SFU()
        t2 = rdi_transforms.TransformSFU_XYZ(0, 0.1919, 0)
        t3 = rdi_transforms.TransformXYZ_BEAM()
        t = t3*t2*t1
        with open('tmp','w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            
            pd0.send_to(t)
            t.send_to(writer)
            
            pd0.process(self.filename)

    def test_transforms(self):
        pd0 = rdi_reader.PD0()

        t = rdi_transforms.TransformENU_SFU()

        with open('tmp5','w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            pd0.send_to(t)
            t.send_to(writer)
            pd0.process(self.filename)

    def test_salinity_correction(self):
        pd0 = rdi_reader.PD0()
        t1 = rdi_transforms.TransformENU_SFU()
        t2 = rdi_transforms.TransformSFU_XYZ(0, 0.1919, 0)
        t3 = rdi_transforms.TransformXYZ_BEAM()
        t = t3*t2*t1

        T1 = rdi_transforms.TransformSFU_ENU()
        T2 = rdi_transforms.TransformXYZ_SFU(0, 0.1919, 0)
        T3 = rdi_transforms.TransformBEAM_XYZ()
        T = T1*T2*T3

        sosc = rdi_corrections.CurrentCorrectionFromSalinity(SA=7)
        
        with open('tmp1','w') as fp:
            with open('tmp2','w') as fp2:
                writer = rdi_writer.AsciiWriter(fp)
                writer2 = rdi_writer.AsciiWriter(fp2)
                # branch to writer and transform:
                pd0.send_to(writer)
                pd0.send_to(t)

                t.send_to(sosc)
                sosc.send_to(T)
                T.send_to(writer2)
                
                pd0.process(self.filename)
            
    def test_aggregator(self):
        pd0 = rdi_reader.PD0()
        agg = rdi_corrections.Aggregator(60)
        with open('tmp3', 'w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            pd0.send_to(agg)
            agg.send_to(writer)
            pd0.process(self.filename)

    def test_attitude_correction(self):
        pd0 = rdi_reader.PD0()
        att = rdi_corrections.AttitudeCorrectionTiltCorrection(0.83, 0, 0, 'rotation')
        with open('tmp4', 'w') as fp:
            writer = rdi_writer.AsciiWriter(fp)
            pd0.send_to(att)
            att.send_to(writer)
            pd0.process(self.filename)

    def test_backscatter(self):
        pd0 = rdi_reader.PD0()
        att = rdi_corrections.AttitudeCorrectionTiltCorrection(0.83, 0, 0, 'rotation')
        backscatter =rdi_backscatter.AcousticCrossSection(S=7, k_t=1e-8, N_t=45, db_per_count=[0.61]*4)
        writer = rdi_writer.NDFWriter('backscatter.ndf')

        pd0.send_to(att)
        att.send_to(backscatter)
        backscatter.send_to(writer)

        pd0.process(self.filename)

    def test_pipeline(self):
        t1 = rdi_transforms.TransformENU_SFU()
        t2 = rdi_transforms.TransformSFU_XYZ(0, 0.1919, 0)
        t3 = rdi_transforms.TransformXYZ_BEAM()
        t = t3*t2*t1

        T1 = rdi_transforms.TransformSFU_ENU()
        T2 = rdi_transforms.TransformXYZ_SFU(0, 0.1919, 0)
        T3 = rdi_transforms.TransformBEAM_XYZ()
        T = T1*T2*T3

        sosc = rdi_corrections.CurrentCorrectionFromSalinity(SA=7)
        writer = rdi_writer.DataStructure()

        pd0 = rdi_reader.PD0()
        pipeline = rdi_reader.make_pipeline(t, sosc, T,writer)
        pd0.send_to(pipeline)
        pd0.process(self.filename)
        print(np.mean(pipeline.data['velocity_east']))
        #should be -6.199 or something

    def test_Rangelimit(self):
        pd0 = rdi_reader.PD0()
        att = rdi_corrections.AttitudeCorrectionTiltCorrection(0.83, 0, 0, 'rotation')
        rl = rdi_qc.VelocityRangeLimit(pitch_mount_angle=11,
                                           XdcrDepth_scale_factor = 10,
                                           qw=0.005**2, qH=0.001**2, rz = 0.15**2, rH=0.20**2)
        writer = rdi_writer.NDFWriter('velocity_range_limit.ndf')
        writer.set_custom_parameter('bottom_track', 'WaterDepth', dtype='scalar')
        with writer:
            pd0.send_to(att)
            att.send_to(rl)
            rl.send_to(writer)
            pd0.process(self.filename)
        
if __name__=="__main__":
    unittest.main()
    if 0:
        t = TestWriter()
        t.setUp()
        t.test_Rangelimit()

        import ndf
        data = ndf.NDF("velocity_range_limit.ndf", open_mode='open')
        tm, dpt = data.get('variable_leader WaterDepth')
