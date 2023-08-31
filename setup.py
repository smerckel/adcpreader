"""

adcpreader: a python module for reading and processing RDI ADCP data files.

"""

import adcpreader
version = adcpreader.__VERSION__

from distutils.core import setup

setup(name="adcpreader",
      version=version,
      packages = ["adcpreader"],
      scripts = [],
      author = "Lucas Merckelbach",
      author_email = "lucas.merckelbach@hereon.de",
      url = "https://github.com/smerckel/adcpreader.git")

