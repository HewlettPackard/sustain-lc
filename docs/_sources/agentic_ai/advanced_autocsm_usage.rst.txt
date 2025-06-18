Another example
===============

This section covers advanced topics like building custom models for Sustain-LC. It requires the following repositories and software:

Software Installation
---------------------

Dymola and OpenModelica both provide a GUI and CLI for creating, compiling, and simulating Modelica models and exporting FMUs.

Repository Installation
-----------------------

Clone these into your working folder so that your Modelica IDE (Dymola/OpenModelica) can see them:

1. **Modelica Buildings library**  
   ``git clone https://github.com/lbl-srg/modelica-buildings.git``  

   The Modelica Buildings Library (LBL) provides HVAC, pump, valve, and heat-exchanger models, validated against measured data for building/data-center thermal simulation.

2. **TRANSFORM Library**  
   ``git clone https://github.com/ORNL-Modelica/TRANSFORM-Library.git``  

   Oak Ridge’s TRANSFORM toolkit models transient thermal-hydraulic systems, with detailed fluid properties and advanced reactor/cooling-loop components.

3. **datacenterCoolingModel**  
   ``git clone https://code.ornl.gov/exadigit/datacenterCoolingModel.git``  

   ORNL’s data-center–specific CSM framework: direct-to-chip, immersion, rear-door heat-exchanger models, and facility-scale coolant loops integrated with IT thermal loads.

4. **AutoCSM**  
   ``git clone https://code.ornl.gov/exadigit/AutoCSM.git``  

   ExaDigiT’s Python framework to auto-generate system-level cooling models: JSON→Modelica→FMU.

Custom Sustain-LC Models Using AutoCSM
--------------------------------------

AutoCSM reads your JSON, injects components from `datacenterCoolingModel`, and writes out a Modelica file. To run:

.. code-block:: bash

   python run_auto_csm.py

Edit `run_auto_csm.py` to point at your JSON and set your solver parameters. The resulting `FrontierModel.fmu` then becomes the backend of your Gymnasium environment (`frontier_env.py`).
