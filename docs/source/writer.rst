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

The typical use of the Writer() class and its subclasses is

::
   
   writer = Writer()

   writer.set_custom_parameter(section, *name, dtype='scalar'|'vector')

   reader.send_to(writer)

The first line creates an instance of the ``Writer`` class, and the
second line allows variables to be written to file, that would be left
out by default. Not all implementations of a Writer subclass honour
this, though. In the third line, the writer is hooked up in the
pipeline. In this case, directly to the reader.
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


AsciiWriter
-----------

The ``AsciiWriter`` class derives from the ``Writer`` class, and it
implements the write_configuration(), write_header() and write_array()
methods. The constructor of the ``AsciiWriter`` class can receive a
file pointer, and using this file pointer to write the data it
receives. If no file pointer is given, then all the data is directed
to stdout.

NetCDFWriter
------------

The ``NetCDFWriter`` class also derives from the ``Writer`` class, and
can be used for writing the data in NetCDF format. Currently the
implementation is basic, and a preset list of variables is written
into the files. The constructor requires the basename of the output
file(s). Optionally a ensemble_size_limit can be set, which limits the
number of ensembles. If more than one file is required, the files are
numbered sequentially.



