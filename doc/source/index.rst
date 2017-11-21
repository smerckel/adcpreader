.. Python RDI ADCP documentation master file, created by
   sphinx-quickstart on Fri Sep  8 17:54:24 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Python RDI ADCP's documentation
==========================================

These pages describe the a python3 implementation of software to read and
process native RDI ADCP binaries. The key idea is to provide a
software to set up a pipeline of operations, from reading the
data to writing the data into the desired output format, where
intermediate steps may involve coordinate transformations, alignment
corrections, speed of sound corrections, and filtering data based on
quality control conditions. Such a pipe line can be set up for routine
processing of incoming data, ad-hoc processing data, or reprocessing
existing data.

Binary ADCP files can be big. The aggregate data of a whole experiment
can be even bigger. Special attention has been paid to memory
efficient handling of the data.

The source code is developed on a linux platform, but it should run on
windows and OSX platforms without modification.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   working_example
   reader
   writer
   corrections
   quality_control
   transformations
   backscatter
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Examples of RST
---------------
   
+--------+----------+------+
| nombre | apellido | edad |
+========+==========+======+
| pepe   | zarate   | 28   |
+--------+----------+------+
| toto   | garcia   | 29   |
+--------+----------+------+

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}


   
:math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`




  
