Making pipelines
================

Defining a pipeline of processes
--------------------------------

If we have a number of processing steps between reading the ensembles
and writing them, we can define a number of operations and set up a
pipeline by sending from one operation to the next. As an example we
can define the following processors ::

  # first transform
  t0 = rdi.rdi_transforms.TransformENU_SFU()
  t1 = rdi.rdi_transforms.TransformSFU_XYZ(0, mounted_pitch, 0)
  t2 = rdi.rdi_transforms.TransformXYZ_BEAM()
  transform_enu_to_beam = t2 * t1 * t0

  # second transform
  t3 = rdi.rdi_transforms.TransformBEAM_XYZ()
  t4 = rdi.rdi_transforms.TransformXYZ_SFU(0, mounted_pitch, 0)
  transform_beam_to_sfu = t4 * t3

  # third transform
  transform_sfu_to_enu = rdi.rdi_transforms.TransformSFU_ENU()

  # some data filtering:
  max_velocity = 0.75
  qc_u_limit = rdi.rdi_qc.ValueLimit()
  qc_u_limit.set_discard_condition('velocity','Velocity1','||>',max_velocity)
  qc_u_limit.set_discard_condition('velocity','Velocity2','||>',max_velocity)
  qc_u_limit.set_discard_condition('velocity','Velocity3','||>',max_velocity)
  qc_u_limit.set_discard_condition('velocity','Velocity4','||>',max_velocity)

  qc_snr_limit = rdi.rdi_qc.SNRLimit(3)

  qc_amplitude_limit = rdi.rdi_qc.AcousticAmplitudeLimit(75)

  # and a writer (sink)
  writer = rdi.rdi_writer.NetCDFWriter('output.nc')

  # and reader (source)
  reader = rdi.rdi_reader.PD0()

Now we can setup the pipeline of operations as follows ::

  reader.send_to(transform_enu_to_beam)
  transform_enu_to_beam.send_to(qc_u_limit)
  qc_u_limit.send_to(qc_snr_limit)
  qc_snr_limit.send_to(qc_amplitude_limit)
  qc_amplitude_limit.send_to(transform_beam_to_sfu)
  transform_beam_to_sfu.send_to(transform_sfu_to_enu)
  transform_sfu_to_enu.send_to(writer)

This kind of notation is awkward to read and involves unnecessary
typing. A short-cut for this pipeline is ::

  pipeline = rdi.rdi_reader.make_pipeline(reader,
                                          transform_enu_to_beam,
					  qc_u_limit,
					  qc_snr_limit,
					  qc_amplitude_limit,
					  transform_beam_to_sfu,
					  transform_sfu_to_enu,
					  writer)

To feed all data through the pipeline, it is sufficient to call ::

  reader.process(filename)

Branching
---------

Splitting a pipeline and feeding one ensemble through two different
pipelines is a walk in the park. Let's say, we want to have the
processed data written to file in different coordinate systems. To
then end we have to define another writer process ::

  writer_sfu = rdi.rdi_writer.NetCDFWriter('output-sfu.nc')

Then we can set up a new common pipeline, and two branches ::
  
  pipeline = rdi.rdi_reader.make_pipeline(reader,
                                          transform_enu_to_beam,
					  qc_u_limit,
					  qc_snr_limit,
					  qc_amplitude_limit)
  # and the two branches:

  branch_enu = rdi.rdi_reader.make_pipeline(pipeline,
                                            transform_beam_to_sfu,
					    transform_sfu_to_enu,
					    writer)
					    
  branch_sfu = rdi.rdi_reader.make_pipeline(pipeline,
                                            transform_beam_to_sfu,
					    writer_sfu)
  # and process them:

  reader.process(filename)

Now we have for given filename a netcdf file with velocities expressed
in ENU coordinates, and in SFU coordinates.

Merging branches
----------------

Suppose we have two different branches, we can also merge them into
one:

.. figure:: figures/merge.svg

	    A schematic representation of two branches merged into
	    one.
	    

A practical example is to add external data to the ensemble data,
where external data can be data collected by a glider, or GPS
positions, for example. A simple way to achieve this is to use the
class :class:`DataFuse`::

    import rdi.rdi_datafuse

    :
    :
    
    data_fuser = rdi.rdi_datafuse.DataFuse("glider_flight")
	    

    pipeline1.send_to(data_fuser)
    pipeline2.send_to(data_fuser)

    data_fuser.send_to(merged_pipeline)

Here, we create a data_fuser object, which combines two streams into
one. The first stream is ``pipeline1``, to which the second stream
will be appended, and subsequently output as ``merged_pipeline``. The
argument to the constructor of the :class:`DataFuse` class, in this case
``"glider_flight"``, denotes the section name into which the data from
stream 2 should be saved. If the section name already exists, then the
data fields from stream 2 are added to the existing section of the
ensemble.

Since order matters, it is important that the primary stream sends its
data to the data_fuser first. In the example given, pipeline1 sends
the data to data_fuser, before pipeline2 and is therefore, the primary
stream. The data coming from pipeline2, will be added to the data from
pipeline1 under the section name ``"glider_flight"``.

.. note::

   It is important to realise that (the coroutine of) the :class:`DataFuser`
   class expects two blobs of input data, to yield one blob of output
   data. That means that every time an ensemble is produced through
   pipeline1, pipeline2 also produces one dictionary with data to add
   to this ensemble. If the input pipelines are not synchronised, they
   must be done so first, before the can be merged by an instance of
   DataFuser.


Synchronising pipelines
_______________________

For an instance of the :class:`DataFuser` class to be able to keep track of
incoming data, it must be ensured that a data blob from the primary
stream is followed by the secondary stream. If the secondary stream is
created by branching a single stream before, this is automatically
ensured, see the figure below:

.. figure:: figures/branch_and_merge.svg

	    An input pipeline is, branched, and then merged again. In
	    this scenario, the pipelines are automatically synchronised.

The situation is different when the data to be merged come from an
external source. In this case the user needs to assure that this
stream is synchronised with the primary stream. Now we also branche
the input pipeline. One branch, the primary pipeline, is connected to
the :class:`DataFuser` instance, whereas the other branche, is connected to
a separate processing unit, see Figure below. This unit is now responsible for
synchronising the external data, that are stored in an external data
file, for example, with the ensemble it received:

.. code:: text
	  
    upon receiving an ensemble do:

    1) read the timestamp of the ensemble
    2) look up in the external data file the required data pertaining
       to this time stamp
    3) output a dictionary with the selected data

An example of a processing unit that does this is the
:class:`rdi.rdi_datafuse.NDFReader` class.

.. figure:: figures/merge_external.svg

	    An example of fusing data from an external source. The
	    input pipeline is branched, as before, with the primary
	    stream directly connected to the DataFuser, and the
	    secondary stream is processed by the External data reader,
	    which is responsible for the synchronisation.
       
