Agentic LLM-Based Digital Twin for Liquid Cooling
=================================================

This section describes a modular agent-based architecture enhanced with Large Language Models (LLMs) to enable **explainable control** in liquid-cooled data centers. The system is built on functional agents that communicate via a centralized **Message Bus**.

.. image:: ../images/llm_agentic_info.png
   :scale: 40%
   :alt: Agentic LLM System Overview
   :align: center

Agent Types
-----------

- **LLM/LLM-supported Agents (purple):** Handle reasoning, planning, and control decisions.
- **Non-LLM Agents (blue):** Handle system-specific monitoring, control, and interfacing.
- **Tools (green):** Provide analytics, modeling, and visualization.

Main Components
---------------

- **Orchestration:** Manages agent lifecycle, communication, and error handling.
- **Agent Monitor:** Tracks performance and suggests optimizations.
- **Maintenance:** Detects anomalies and enables predictive maintenance.
- **Math Toolbox:** Offers modeling tools, statistical reasoning, and uncertainty analysis.
- **Configuration:** Handles system parameters, logic, and hyperparameter tuning.
- **Visualization:** Supports explanations, dashboards, and trend displays.
- **Control:** Issues real-time actions, combines RL and LLM-based policies.
- **Sensor:** Interfaces with the physical system and digital twin.
- **LLM Control Agents:** Issue adaptive actions for cooling elements based on historical and real-time data.
- **Liquid Cooling System:** Uses a CDU and cabinet valves for precise thermal control, based on sensor feedback (T_in, T_out).

Use this architecture to combine physical system modeling, real-time control, and explainable AI for advanced cooling management.


.. toctree::
   :maxdepth: 2

   agentic2
   agentic3
