import rdi.rdi_reader, rdi.rdi_writer, rdi.rdi_qc, rdi.rdi_transforms, rdi.rdi_corrections, rdi.rdi_backscatter


import unittest


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.filename = "../data/PF230519.PD0"

    def tearDown(self):
        pass

    def test_ascii(self):
        pd0 = rdi.rdi_reader.PD0()
        with open('PF230519.txt', 'w') as fp:
            writer = rdi.rdi_writer.AsciiWriter(fp)
            pd0.send_to(writer)
            pd0.process(self.filename)

    def test_ndf(self):
        pd0 = rdi.rdi_reader.PD0()
        writer = rdi.rdi_writer.NDFWriter('PF230519.ndf')
        pd0.send_to(writer)
        pd0.process(self.filename)

            
    def test_netcdf(self):
        pd0 = rdi.rdi_reader.PD0()
        writer = rdi.rdi_writer.NetCDFWriter('PF230519.nc')
        pd0.send_to(writer)
        pd0.process(self.filename)

    def test_qc(self):
        pd0 = rdi.rdi_reader.PD0()
        
        qc = rdi.rdi_qc.ValueLimit()
        qc.set_discard_condition('velocity', 'Velocity1', '||>', 1)
        with open('tmp','w') as fp:
            writer = rdi.rdi_writer.AsciiWriter(fp)
            pd0.send_to(qc)
            qc.send_to(writer)
            pd0.process(self.filename)
        
    def test_transforms_to_beams(self):
        pd0 = rdi.rdi_reader.PD0()

        t1 = rdi.rdi_transforms.TransformENU_SFU()
        t2 = rdi.rdi_transforms.TransformSFU_XYZ(0, 0.1919, 0)
        t3 = rdi.rdi_transforms.TransformXYZ_BEAM()
        t = t3*t2*t1
        with open('tmp','w') as fp:
            writer = rdi.rdi_writer.AsciiWriter(fp)
            
            pd0.send_to(t)
            t.send_to(writer)
            
            pd0.process(self.filename)

    def test_transforms(self):
        pd0 = rdi.rdi_reader.PD0()

        t = rdi.rdi_transforms.TransformENU_SFU()
        
        writer = rdi.rdi_writer.AsciiWriter()
            
        pd0.send_to(t)
        t.send_to(writer)
            
        pd0.process(self.filename)

    def test_salinity_correction(self):
        pd0 = rdi.rdi_reader.PD0()
        t1 = rdi.rdi_transforms.TransformENU_SFU()
        t2 = rdi.rdi_transforms.TransformSFU_XYZ(0, 0.1919, 0)
        t3 = rdi.rdi_transforms.TransformXYZ_BEAM()
        t = t3*t2*t1

        T1 = rdi.rdi_transforms.TransformSFU_ENU()
        T2 = rdi.rdi_transforms.TransformXYZ_SFU(0, 0.1919, 0)
        T3 = rdi.rdi_transforms.TransformBEAM_XYZ()
        T = T1*T2*T3

        sosc = rdi.rdi_corrections.CurrentCorrectionFromSalinity(SA=7)
        
        with open('tmp1','w') as fp:
            with open('tmp2','w') as fp2:
                writer = rdi.rdi_writer.AsciiWriter(fp)
                writer2 = rdi.rdi_writer.AsciiWriter(fp2)
                # branch to writer and transform:
                pd0.send_to(writer)
                pd0.send_to(t)

                t.send_to(sosc)
                sosc.send_to(T)
                T.send_to(writer2)
                
                pd0.process(self.filename)
            
    def test_aggregator(self):
        pd0 = rdi.rdi_reader.PD0()
        agg = rdi.rdi_corrections.Aggregator(60)
        with open('tmp3', 'w') as fp:
            writer = rdi.rdi_writer.AsciiWriter(fp)
            pd0.send_to(agg)
            agg.send_to(writer)
            pd0.process(self.filename)

    def test_attitude_correction(self):
        pd0 = rdi.rdi_reader.PD0()
        att = rdi.rdi_corrections.AttitudeCorrectionTiltCorrection(0.83, 0, 0, 'rotation')
        with open('tmp4', 'w') as fp:
            writer = rdi.rdi_writer.AsciiWriter(fp)
            pd0.send_to(att)
            att.send_to(writer)
            pd0.process(self.filename)

    def test_backscatter(self):
        pd0 = rdi.rdi_reader.PD0()
        att = rdi.rdi_corrections.AttitudeCorrectionTiltCorrection(0.83, 0, 0, 'rotation')
        backscatter =rdi.rdi_backscatter.AcousticCrossSection(S=7, k_t=1e-8, N_t=45, db_per_count=[0.61]*4)
        writer = rdi.rdi_writer.NDFWriter('backscatter.ndf')

        pd0.send_to(att)
        att.send_to(backscatter)
        backscatter.send_to(writer)

        pd0.process(self.filename)

        
if __name__=="__main__":
    unittest.main()        


