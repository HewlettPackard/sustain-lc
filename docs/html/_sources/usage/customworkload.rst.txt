.. _custom-workload-data:

Custom Workload Data
====================

By default, Green-DCC includes workload traces from `Alibaba <https://github.com/alibaba/clusterdata>`_ and `Google <https://github.com/google/cluster-data>`_ data centers. These traces are used to simulate the tasks that the datacenter needs to process, providing a realistic and dynamic workload for benchmarking purposes.

Data Source
-----------

The default workload traces are extracted from:

- **Alibaba 2017 CPU Data** (`LINK <https://github.com/alibaba/clusterdata>`_)  
- **Google 2011 CPU Data** (`LINK <https://github.com/google/cluster-data>`_)  

Expected File Format
--------------------

Workload trace files should be in ``.csv`` format, with two columns:

1. A timestamp or index (must be unnamed)  
2. The corresponding DC utilization (``cpu_load``)

The CPU load must be expressed as a fraction of the DC utilization (between 0 and 1). The workload file must contain one year of data with an hourly periodicity (365 × 24 = 8760 rows).

Example Workload Trace File
---------------------------

.. code-block:: csv

   ,cpu_load
   1,0.380
   2,0.434
   3,0.402
   4,0.485
   ...

Integration Steps
-----------------

- Place the new workload trace file in the ``data/Workload`` folder.  
- Update the ``workload_file`` entry in the ``DEFAULT_CONFIG`` dictionary in ``hierarchical_env.py`` with the path to the new workload trace file.  
