"""

rdi: a python module for reading and processing RDI ADCP data files.

"""

import rdi
version = rdi.__VERSION__

from distutils.core import setup

setup(name="rdi",
      version=version,
      packages = ["rdi"],
      scripts = ['pd0rename.py'],
      author = "Lucas Merckelbach",
      author_email = "lucas.merckelbach@hzg.de",
      url = "")

