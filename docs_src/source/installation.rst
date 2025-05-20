Installation Guide for Sustain-LC
=================================

This document shows you how to get Sustain-LC up and running.

Prerequisites
-------------
- Python 3.10 (via Conda)
- Git  
- Conda  
- A Unix-style shell (bash, zsh) or PowerShell on Windows

Clone the repository
--------------------
First, grab the code:

.. code-block:: bash

   git clone https://github.com/HewlettPackard/sustain-lc.git
   cd sustain-lc

Create and activate a virtual environment
-----------------------------------------
Using Conda:

.. code-block:: bash

   # Create a new conda env named "sustain-lc" with Python 3.10
   conda create -n sustain-lc python=3.10
   conda activate sustain-lc

Install Python dependencies
---------------------------
All runtime requirements are listed in `requirements.txt`. Install them with:

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

Install PyFMI
-------------
Install PyFMI using conda:

.. code-block:: bash

   conda install -c conda-forge pyfmi

(Optional) Install Sphinx for docs
----------------------------------
If you plan to build the docs yourself, add the following to your `requirements.txt`:

.. code-block:: text

    sphinx>=6.0
    furo
    sphinx-autodoc-typehints

Then install:

.. code-block:: bash

   pip install -r requirements.txt

Build the documentation
-----------------------
From the project root:

.. code-block:: bash

   cd docs
   make html

This will generate HTML under ``docs/build/html``. You can then open it:

.. code-block:: bash

   # macOS
   open docs/build/html/index.html

   # Linux
   xdg-open docs/build/html/index.html

   # Windows (PowerShell)
   start docs\build\html\index.html

That's it! You now have Sustain-LC installed and its documentation built locally.
