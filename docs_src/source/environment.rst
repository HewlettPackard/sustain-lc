Environment Implementation
==========================

The environment interfaces with a high-fidelity Modelica model compiled as a Functional Mock-up Unit (FMU) version 2.0 for Co-Simulation, leveraging the ``PyFMI`` library for interaction.

FMU Integration and Simulation Core
-----------------------------------

The core of the simulation is a Modelica model representing the data center’s liquid cooling thermodynamics. This model is compiled into an FMU, for example ``LC_Frontier_5Cabinet_4_17_25.fmu``. The environment utilizes ``PyFMI`` to:

1. Load the FMU and parse its model description.  
2. Instantiate the FMU for simulation.  
3. Set up the experiment parameters, including start time (0.0) and a tolerance (if specified, default is FMU’s choice).  
4. Initialize the FMU into its starting state.  
5. During an episode step:  
   - Set input values (actions from the RL agent) to specified FMU variables.  
   - Advance the simulation time by ``sim_time_step`` using the ``fmu.do_step()`` method. This is repeated until the agent’s ``step_size`` is covered.  
   - Get output values (observations for the RL agent and values for reward calculation) from specified FMU variables.  
6. Terminate and free the FMU instance upon closing the environment or resetting for a new episode.

The FMU variable names used for interfacing are explicitly defined within the environment:

- **Action Variables:** ``self.fmu_action_vars`` (e.g., ``pump1.speed_in``, ``valve1.position_in``)  
- **Observation Variables:** ``self.fmu_observation_vars`` (e.g., ``serverRack1.T_out``, ambient.T)  
- **Power Consumption Variables (for reward):** ``self.fmu_power_vars`` (e.g., [pump1.P, fan1.P])  
- **Target Temperature Variable (for reward):** ``self.fmu_target_temp_var`` (e.g., ``controller.T_setpoint``)  

State (Observation) Space
-------------------------

The observation space is defined as a continuous ``gymnasium.spaces.Box`` with specific lower and upper bounds. It comprises the following variables retrieved from the FMU:

- ``FrontierNode.AvgBladeGroupTemp``: Average temperature of a Blade Group in a cabinet (K).  
- ``FrontierNode.AvgBladeGroupPower``: Average power input to each Blade Group in a cabinet (W).  

The bounds for these observations are set to, for example, 273.15 K and 373.15 K for temperature measurements, and between 0.0 kW and 400 kW for power input measurements.

For the Cooling Tower Markov Decision Process, we have a similar observation space:

- ``FrontierNode.CoolingTower.CellPower``: Average power consumption of each cell of the cooling tower (W).  
- ``FrontierNode.CoolingTower.WaterLeavingTemp``: Average temperature of the water leaving each cooling tower (K).  
- ``T_owb``: Outside air wet-bulb temperature.  

Action Space
------------

The action space is a hybrid of continuous ``gymnasium.spaces.Box`` and discrete ``gymnasium.spaces.Discrete``, allowing the agent to control:

- ``FrontierNode.CDU.Pump.normalized_speed``: Scaled speed of the CDU pump (–1 to 1).  
- ``FrontierNode.CDU.TempSetpoint``: Scaled coolant supply temperature setpoint (–1 to 1).  
- ``FrontierNode.CDU.AvgBladeGroupValve``: Scaled valve opening to allow coolant to collect heat from the corresponding blade group (–1 to 1).  
- ``FrontierNode.CoolingTower.WaterLvTSPT``: Discrete setting of cooling tower water leaving temperature setpoint delta.  

These scaled values allow the neural network models used for the RL agents to learn properly and not saturate at the activation layers.

Reward Function
---------------

The reward function guides the RL agent towards desired operational states. It is calculated at each step as:

.. math::

   R_{\text{blade}} = - \sum_{i,j} T_{i,j}

which is the negative of the aggregate temperature of the blade groups, and

.. math::

   R_{\text{cooling tower}} = - \sum_{i,j} P_{i,j}

which is the negative of the total cooling tower power consumption at each time step.

Where:

- \(T_{i,j}\) is the temperature of the \(j\)th blade group of the \(i\)th cabinet, \(\forall j \in 1 \ldots B\) and \(\forall i \in 1 \ldots C\).  
- \(P_{i,j}\) is the power consumption of the \(j\)th cell of the \(i\)th cooling tower, \(\forall j \in 1 \ldots m\) and \(\forall i \in 1 \ldots N\).  

The goal is to minimize server temperatures below the target and minimize energy consumption.

Episode Dynamics and Simulation Control
---------------------------------------

An episode runs for a maximum of ``max_episode_duration``. The agent interacts with the environment at discrete time intervals defined by ``step_size``. For each agent step, the FMU’s ``do_step()`` method is called ``step_size / sim_time_step`` times. The ``reset()`` method terminates the current FMU instance, re-instantiates and re-initializes it, ensuring a consistent starting state for each new episode. Initial observations are drawn from the FMU after initialization.
