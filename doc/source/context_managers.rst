Reading from and writing to multiple files: the use of context managers
=======================================================================


The method ``process()`` of the `PD0` class closes by default the
pipeline when the last ensemble has been processed. This ensures that
output files can be closed. If we want to use the pipeline again for a
different set of files, we would need to redefine the pipeline. This
behaviour can be altered by supplying the argument
``close_coroutines_at_exit=False`` to the ``process()`` method
as in ::
   
   reader.process(fn, close_coroutines_at_exit=False)

The consequence is that we have to bother ourselves when to close a
file. To this end, we can use context managers. For example ::

  for i, fn in enumerate(pd0_filenames):
      # create a matching output filename:
      output_filename = os.path.join(output_dir, os.path.basename(fn))
      writer.output_file = output_filename.replace("PD0", "nc")

      # use the writer's context manager to take care of opening and
      #closing files.
      
      with writer:
          reader.process(fn, close_coroutines_at_exit=False)


As we don't want to leave an open pipeline, when we have done the
processing of all files, we would need to tell the reader to
close it ::

  reader.close_coroutine()

Instead of calling the ``close_coroutine`` method ourselves, we can
also leave it up to the context manager of the reader, and our code
reads ::

  with reader:
      for i, fn in enumerate(pd0_filenames):
          # create a matching output filename:
	  output_filename = os.path.join(output_dir, os.path.basename(fn))
	  writer.output_file = output_filename.replace("PD0", "nc")

	  # use the writer's context manager to take care of opening and
	  # closing files.

	  with writer:
	      reader.process(fn, close_coroutines_at_exit=False)











   
