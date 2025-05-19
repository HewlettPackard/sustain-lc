Green-DCC Documentation
=======================

Green-DCC is a benchmark environment designed to evaluate dynamic workload distribution techniques for sustainable Data Center Clusters (DCC). It aims to reduce the environmental impact of cloud computing by distributing workloads within a DCC that spans multiple geographical locations. The benchmark environment supports the evaluation of various control algorithms, including reinforcement learning-based approaches.

This page contains the documentation for the GitHub `repository <https://github.com/HewlettPackard/green-dcc>`_ and the paper  
“Green-DCC: Benchmarking Dynamic Workload Distribution Techniques for Sustainable Data Center Cluster”  
<https://openreview.net/forum?id=DoDz1IXjDB>_

.. image:: images/hier.png
   :scale: 18 %
   :alt: Green-DCC Framework for Data Center Cluster Management
   :align: center

Demo of Green-DCC
-----------------

A demo of Green-DCC is given in the Google Colab notebook below:

.. image:: images/colab-badge.png
   :alt: Google Colab
   :target: https://colab.research.google.com/drive/1NdU2-FMWxEXN2dPM1T9MSVww5jpnFExP?usp=sharing

Key features of Green-DCC
--------------------------

- Dynamic geographic distribution of workloads across data centers in a cluster, driven by a central Top-Level Agent.
- Dynamic temporal scheduling decisions by the Top-Level Agent, determining whether tasks should be executed immediately or deferred to optimize efficiency.
- Incorporation of non-uniform computing resources, cooling capabilities, auxiliary power resources, and varying external weather and carbon intensity conditions.
- A dynamic bandwidth cost model that accounts for the geographical characteristics and amount of data transferred.
- Realistic workload execution delays to reflect changes in data center capacity and demand.
- Support for benchmarking multiple heuristic and reinforcement learning-based approaches.
- Customizability to address specific needs of cloud providers or enterprise data center clusters.

Green-DCC provides a complex, interdependent, and realistic benchmarking environment that is well-suited for evaluating reinforcement learning algorithms applied to data center control. The ultimate goal is to optimize workload distribution to minimize the carbon footprint, energy usage, and energy cost, while considering various operational constraints and environmental factors.

.. image:: images/GreenDCCv3.png
   :scale: 15 %
   :alt: Green-DCC Framework demonstrating Geographic and Temporal Scheduling Strategies
   :align: center

Scheduling Strategy
-------------------

Green-DCC centralizes both geographic and temporal workload distribution decisions in the Top-Level Agent. At each scheduling step, the agent determines whether to execute a given task immediately or defer it, and selects the optimal data center for execution. This unified spatio-temporal approach ensures that all load-shifting logic is handled globally.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   gettingstarted
   overview/index
   usage/index
   evaluation
   code/modules
   contribution_guidelines
   references
