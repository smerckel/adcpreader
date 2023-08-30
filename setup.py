"""

rdi: a python module for reading and processing RDI ADCP data files.

"""

import adcpreader
version = adcpreader.__VERSION__

from distutils.core import setup

setup(name="adcpreader",
      version=version,
      packages = ["adcpreader", "slocum"],
      scripts = ['pd0rename.py'],
      author = "Lucas Merckelbach",
      author_email = "lucas.merckelbach@hereon.de",
      url = "")

