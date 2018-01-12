Writer
======

As processing ADCP data is rather costly, it is assumed that at the
end of the processing pipe line a "sink" is implemented that consumes
all (processed) data and writes it to one or more files. Indeed, the
sole use of this software may be to convert the raw binary files into
easily accessible files in a known format. To facilitate this, a
Writer() class is defined that implements the required machinery to
hook up a writer to the pipe line.

The Writer() class itself does not
know about any specifications of an output format. To implement code
that writes NetCDF files, for example, a class is to be created that
is subclassed from the Writer() class and implements the output format
specific calls.

The API of the Writer() class and its subclasses is

::
   
   W = Writer()

   W.write_ensembles(ensembles)

   W.set_custom_parameter(section, *name, dtype='scalar'|'vector')

   

The method W.write_ensembles(ensembles) is effectively the sink method
and should be last statement in the pipeline. Instead of calling
W.write_ensembles(ensembles), the instance W can be called with the
ensembles as argument.

The Writer() class defines a further three methods:

::
   
   W.write_configuration(config, fd)

   W.write_header(config, fd)

   W.write_array(config, data1d, data2d, fd)

but these are not implemented, and should be implemented by the
subclasses. These methods are specific to the output format required.

When the W.write_ensembles() method is called, then the methods

::
   
   W.write_configuration(config, fd)

   W.write_header(config, fd)

are called for the first ensemble only. The parameters to both these
methods are config and fd. Config is the dictionary of the
fixed_leader data block, and fd is a file descriptor set by
self.output_file (defaults to None). Depending on the requirements of
the user (and the specifications of the output file format) a subclass
of Writer() should implement these methods accordingly. Also, it is
the responsibility of the subclass to provide a convenient way to
provide an output filename or output filename template.

The subclasses of Writer() should also implement a third method

::

   W.write_array(config, data1d, data2d, fd)

which is called for every ensemble that is read. The parameter config
is a before, as well is the parameter fd. The data structures data1d,
and data2d are one and two dimensional data structures
respectively. Both data structures are dictionaries where the key
refers to the variable name. The value is a list, which gets cleared
after processing each ensemble. (This may change in the future, to
write to file only when a specified number of ensembles have been
processed.)

btv1  BTVel1
btv2  BTVel2
btv3  BTVel3
btv4' BTVel4
btpg1 PG1
btpg2 PG2
btpg3 PG3
btpg4 PG4


   






The remaining three methods are not implemented
by the Writer() class and should be implemented by the subclasses,  as
these particulare 

