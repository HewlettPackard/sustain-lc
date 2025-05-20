Custom Carbon Intensity Data
============================

Carbon Intensity (CI) data represents the carbon emissions associated with electricity consumption.  
Green-DCC includes CI data files for various locations to simulate the carbon footprint of the DC’s energy usage.

Data Source
-----------

The default carbon intensity data files are extracted from:

- U.S. Energy Information Administration API (`EBA bulk download`_)  

.. _EBA bulk download: https://api.eia.gov/bulk/EBA.zip

Expected File Format
--------------------

Carbon intensity files should be in ``.csv`` format, with columns representing different energy sources and the average carbon intensity.  The columns typically include:

- ``timestamp``: The time of the data entry
- ``WND``: Wind energy
- ``SUN``: Solar energy
- ``WAT``: Hydropower energy
- ``OIL``: Oil energy
- ``NG``: Natural gas energy
- ``COL``: Coal energy
- ``NUC``: Nuclear energy
- ``OTH``: Other energy sources
- ``avg_CI``: The average carbon intensity

Example Carbon Intensity File
-----------------------------

.. code-block:: csv

   timestamp,      WND, SUN, WAT, OIL,   NG,  COL,  NUC, OTH, avg_CI
   2022-01-01T00:00:00+00:00, 1251,   0, 3209,   0, 15117, 2365, 4992, 337, 367.450
   2022-01-01T01:00:00+00:00, 1270,   0, 3022,   0, 15035, 2013, 4993, 311, 363.434
   2022-01-01T02:00:00+00:00, 1315,   0, 2636,   0, 14304, 2129, 4990, 312, 367.225
   2022-01-01T03:00:00+00:00, 1349,   0, 2325,   0, 13840, 2334, 4986, 320, 373.228
   ...

Integration Steps
-----------------

- Add your new carbon intensity CSV files to the ``data/CarbonIntensity/`` folder.
- In ``hierarchical_env.py``, update the ``DEFAULT_CONFIG['cintensity_file']`` entry to point at your new file.
