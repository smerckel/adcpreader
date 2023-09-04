|PyPI version| |Docs badge| |License|

ADCPREADER - A python3 module for reading RDI's ADCP binary data files.
=======================================================================

Change log
----------

Version 0.2.1

* Changed name to adcpreader
* Updated documentation
* Documentation on readthedocs
* Glider specific parts of the code have been removed (including unpublished python dependencies)

Version 0.1.0

* Initial release


Synopsis
--------

This python module is primarily intended to read the binary data files
created by RDI's 600 kHz Doppler Velocity Log mounted on Slocum ocean
gliders. The module can, however, also be used to read binary date
from other stand-alone ADCPs, that adhere to RDI's binary data format.

The philosophy behind the implementation of *adcpreader* is that acoustic
ping (ensembles) are processed according to a user-defined
pipeline. Since binary data files can be huge, and the total amount of
data of a deployment even larger, possible issues with limited memory
are dealt with by pushing ensemble per ensemble through the pipeline,
making extensively use of coroutines.

Installation
------------

The python module *adcpreader* can be installed from source, using the
standard method to install python code. Alternatively, *adcpreader* can also
be installed from PyPi, using ``pip install adcpreader``.



Documentation
-------------

Comprehensive documentation is provided at https://adcpreader.readthedocs.io/en/latest/

Quick-start
-----------

For the impatient...

The module *adcpreader* implements a class PD0(), which returns an object the
serves as the source of the pipeline. Usually the end of the pipeline
will be some sink that either writes the data into a file, or into an
object that allows access to the data during an interactive python
session.

In the simplest case we can construct a pipeline with a source and
sink only::

  >>> from adcpreader.rdi_reader import PD0
  >>> from adcpreader.rdi_writer import DataStructure
  >>> source = PD0()
  >>> sink = DataStructure()
  >>> pipeline = source | sink


In the code example above, we create a source operation and a sink
operation, and construct a pipeline using the pipe symbol "|".

Now, we can push data of file *sample.PD0* through the pipeline::

  >>> pipeline.process("sample.PD0")

which results in the sink to contain the data of this file. You can
use :code:`sink.keys()` to list all variables that are accessible
through this object. For example the ensemble numbers can be accesed
as::

  >>> sink.data['Ensnum']


or more compact::

  >>> sink.Ensnum


In this example, we processed in a single file. We could also provide
a list of filenames as argument to :code:`pipeline.process`. However,
we can use the pipeline only once. That is, this will fail::

  >>> pipeline.process("sample.PD0")
  >>> pipeline.process("another_sample.PD0")


This is because under the hood generators and coroutines are
used. When the generator (source) is exhausted, the coroutines are
closed, and cannot be used anymore. Either, all data files are
processed when supplied as a list to :code:`pipeline.process()`, or
the pipeline is defined again.

A third way (not recommended), is to leave the coroutines open, by
supplying the optional keyword
:code:`close_coroutines_at_exit=False`. Then it is the user's
responsibility to close the routines when the pipeline is
invoked for the last time.

An extensive number of operations are defined that can be placed in
the pipeline. Some are for information purposes only, but most will in
some way modify the data. You could define an operator::

  >>> info = adcpreader.rdi_writer.Info(pause=True)

  
and create a new pipeline::

  >>> pipeline = source | info | sink


This will have no effect on the contents of :code:`sink`, but it will
display some information to the terminal (and pause before
continuing).

Other operations will affect the data. Examples, are corrections,
rotations, coordinate transforms, and quality checks. See for the
documentation for further information on https://adcpreader.readthedocs.io/en/latest/.


.. |PyPI version| image:: https://badgen.net/pypi/v/adcpreader
   :target: https://pypi.org/project/adcpreader
.. |Docs badge| image:: https://readthedocs.org/projects/adcpreader/badge/?version=latest
   :target: https://adcpreader.readthedocs.io/en/latest/
.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
