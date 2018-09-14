Pipelines
=========

Generally, the data read from the PD0 files will be further processed in a pipeline style.
The main idea is that a source generator feeds its data through a pipeline, until the
data are consumed by a sink::
  
    source -> op1 -> op2 -> op3 ... -> opn -> sink
              |__________________________|  
                 pipeline of operations

Any number of operations can be added to the pipeline. An example is

* read ensembles for a PD0 file (source)
* apply a coordinate transformation (op 1)
* filter out any ensembles for which no bottom track is available (op2)
* write data into a data structure (sink)

One convenient way to deal with this work flow is to define a pipeline
first::

  from rdi_reader import Pipeline
  from rdi_transforms import TransformENU_FSU
  from rdi_qc import ValueLimit
  
  pipeline = Pipeline()

  vl = ValueLimit(drop_masked_ensembles=True)
  vl.set_discard_condition(section='bottom_track',
                                    parameter = 'BTVel1', 
                                    operator = '||>',
				    value = 0.75)
  pipeline.add(vl)
  
  transform = TransformENU_FSU()
  pipeline.add(transform)

  for ens in pipeline(dvl_filenames = 'a7ff03ed.PD0'):
      print(ens['Ensnum'])

The source is assumed to be an ensemble generator
PD0.ensemble_generator(), which is automatically invoked when calling
the pipeline. The argument (a string or list of strings) supplied is
passed to the PD0.ensemble_generator() method.

