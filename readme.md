<div style="text-align: center;">
    <img src="assets/sustain_lc_logo.png" alt="Logo" width="200"/>
</div>

For more information, please visit the [documentation webpage](https://hewlettpackard.github.io/sustain-lc/).

# SustainLC: Benchmark for End-to-End Control of Liquid Cooling in Sustainable Data Centers

## Quickstart evaluation on collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HewlettPackard/sustain-lc/blob/main/evaluate_mh_ma_ca_ppo.ipynb)

## 🔧 Liquid Cooling Control Benchmark for HPC Data Centers

This Python-based benchmark environment enables the development and evaluation of control strategies for high-performance computing (HPC) data centers, using a digital twin of ORNL’s Frontier supercomputer. It supports energy-efficient cooling optimization across fine-grained server Blade-Groups, Cooling Towers (CT), and Heat Reuse Units (HRU).

## Features

- End-to-end customizable, scalable, and multi-agent control setups
- Real-time control interfaces via Gymnasium and ASHRAE G36-based traditional controllers
- Granular CDU agent control over inlet setpoints, pump flow, and valve positions
- Hybrid and interpretable policy support using decision tree distillation techniques ([Bastani et al., 2018])
- Built-in tools for ablation studies, scalability testing, and LLM-based explainability

Ideal for researchers, engineers, and practitioners in RL, control systems, and sustainable computing infrastructure.

## Installation

### Prerequisites
- Python 3.10+
- Conda (for creating virtual environments)
- Dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/HewlettPackard/sustain-lc.git
    ```
    
2. Navigate to the repository directory:
    ```bash
    cd sustain-lc
    ```

3. Create a new Conda environment with from .YAML file:
    ```bash
    conda env create -f environment.yml
    ```

4. Activate the newly created environment:
    ```bash
    conda activate sustain-lc
    ```

## Quick Start Guide

1. **Train Example:**

To train agent for centalized action policies, run

```bash
   python train_multiagent_ca_ppo.py  
```

To train agent for centalized action multihead policies, run
```bash
   python train_mh_ma_ca_ppo.py  
```

2. **Monitor training on Tensorboard**
   ```bash
   tensorboard --logdir /runs/
   ```

3. **Evaluation Example:**

We provide a more user-friendly example to evaluate the agents via jupyter notebooks. Interested users, may also simply export the notebook to python script and run the resulting file

For evaluating the centalized action policies, users may run the `evaluate_ma_ca_ppo.ipynb` and for multihead policies, they may run `evaluate_mh_ma_ca_ppo copy.ipynb`.


4. **Policy Distillation using Decision Trees (VIPER)**
To distill the policies for the pretrained agents, the users may run the `policy_distillation.ipynb` notebook






## Sustain-LC environment implementation

The environment interfaces with a high-fidelity Modelica model compiled as a Functional Mock-up Unit (FMU) version 2.0 for Co-Simulation, leveraging the PyFMI library for interaction.

### FMU Integration and Simulation Core

The core of the simulation is a Modelica model representing the data center's liquid cooling thermodynamics. This model is compiled into an FMU, for example `LC_Frontier_5Cabinet_4_17_25.fmu`. The environment utilizes PyFMI to:

1.  Load the FMU and parse its model description.
2.  Instantiate the FMU for simulation.
3.  Set up the experiment parameters, including start time (0.0) and a tolerance (if specified, default is FMU's choice).
4.  Initialize the FMU into its starting state.
5.  During an episode step:
    *   Set input values (actions from the RL agent) to specified FMU variables.
    *   Advance the simulation time by `sim_time_step` using the `fmu.do_step()` method. This is repeated until the agent's `step_size` is covered.
    *   Get output values (observations for the RL agent and values for reward calculation) from specified FMU variables.
6.  Terminate and free the FMU instance upon closing the environment or resetting for a new episode.

The FMU variable names used for interfacing are explicitly defined within the environment:

*   **Action Variables:** `self.fmu_action_vars` (e.g., `pump1.speed_in`, `valve1.position_in`)
*   **Observation Variables:** `self.fmu_observation_vars` (e.g., `serverRack1.T_out`, `ambient.T`)
*   **Power Consumption Variables (for reward):** `self.fmu_power_vars` (e.g., `[pump1.P, fan1.P]`)
*   **Target Temperature Variable (for reward):** `self.fmu_target_temp_var` (e.g., `controller.T_setpoint`)

### State (Observation) Space

The observation space is defined as a continuous `gymnasium.spaces.Box` with specific lower and upper bounds. It comprises the following variables retrieved from the FMU:

*   `FrontierNode.AvgBladeGroupTemp`: Average temperature of a Blade Group in a cabinet (K).
*   `FrontierNode.AvgBladeGroupPower`: Average power input to each Blade Group in a cabinet (W).

The bounds for these observations are set to e.g., `273.15 K` and e.g., `373.15 K` for temperature measurements, and between e.g., `0.0 kW` and e.g., `400 kW`, for power input measurements.

For the Cooling Tower Markov Decision Process, we have a similar observation space:

*   `FrontierNode.CoolingTower.CellPower`: Average power consumption of each cell of the cooling tower (W).
*   `FrontierNode.CoolingTower.WaterLeavingTemp`: Average temperature of the water leaving each cooling tower (K).
*   `T_owb`: Outside air wetbulb temperature.

### Action Space

The action space is a hybrid of continuous `gymnasium.spaces.Box` and discrete `gymnasium.spaces.Discrete`, allowing the agent to control:

*   `FrontierNode.CDU.Pump.normalized_speed`: Scaled speed of the CDU pump (-1 to 1).
*   `FrontierNode.CDU.TempSetpoint`: Scaled Coolant supply temperature setpoint (-1 to 1).
*   `FrontierNode.CDU.AvgBladeGroupValve`: Scaled Valve opening to allow coolant to collect heat from the corresponding blade group (-1 to 1).
*   `FrontierNode.CoolingTower.WaterLvTSPT`: Discrete setting of cooling tower water leaving temperature setpoint delta.

These scaled values allow the neural network models used for the RL agents to learn properly and not saturate at the activation layers.

### Reward Function

The reward function guides the RL agent towards desired operational states. It is calculated at each step as:

$$R_{\text{blade}} = \sum_{i,j} T_{i,j}$$

which is the negative of the aggregate temperature of the blade groups

$$R_{\text{coolingtower}} = - \sum_{i,j} P_{i,j}$$

which is the negative of the total cooling tower power consumption at each time step Where:

*   `T_ij` is the temperature of the j<sup>th</sup> blade group of the i<sup>th</sup> cabinet ∀ j in 1...B and ∀ i in 1...C
*   `P_ij` is the power consumption of the j<sup>th</sup> cell of the i<sup>th</sup> cooling tower ∀ j in 1...m and ∀ i in 1...N

The goal is to minimize server temperatures below the target and minimize energy consumption.

### Episode Dynamics and Simulation Control

An episode runs for a maximum of `max_episode_duration`. The agent interacts with the environment at discrete time intervals defined by `step_size`. For each agent step, the FMU's `do_step()` method is called `step_size` / `sim_time_step` times. The `reset()` method terminates the current FMU instance, re-instantiates and re-initializes it, ensuring a consistent starting state for each new episode. Initial observations are drawn from the FMU after initialization.

## Sustain-LC Training Scripts Documentation

*   **Script:** `train_mh_ma_ca_ppo.py`

    This script is designed to train a Proximal Policy Optimization (PPO) agent in an environment that involves multiple agents and components. It can be configured to use Multi-Head (MH), Centralized Action (CA), and Multi-Agent (MA) features, allowing for flexible and expressive policy representations. CA implies there is a shared policy for multiple homogeneous agents within the specified environment.

    **Basic Run Command:** `python train_mh_ma_ca_ppo.py`

    **Key Configurable Parameters**

    The script uses has the following relevant parameters you can modify:

    *   `-exp-name` (str, default: `ppo_ma_ca`): Name for the experiment, used for logging.
    *   `-seed` (int, default: `123`): Random seed for reproducibility.
    *   `-cuda` (flag, default: `True`): Enables CUDA for GPU acceleration if available. Set `-cuda False` to force CPU.
    *   `-env_name` (str, default: `MH_SmallFrontierModel`): Name of the environment.
    *   `-agent_type` (str, default: `MultiHead_CA_PPO`): Type of the RL Agent.
    *   `-max_training_timesteps` (int, default: `5e6`): Total budget for training.
    *   `-max_ep_len` (int, default: `200`): Maximum episode length.
    *   `-lr_actor` (float, default: `3e-4`): Learning rate for the actor optimizer.
    *   `-lr_critic` (float, default: `1e-3`): Learning rate for the critic optimizer.
    *   `-K_epochs` (float, default: `50`): Epochs of training to run for each update.
    *   `-eps_clip` (float, default: `0.2`): clip parameter for PPO.
    *   `-num_centralized_actions` (int, default: `4`): Number of centralized actions for each environment.
    *   `-gamma` (float, default: `0.80`): Discount factor for future rewards.
    *   `-gae_lambda` (float, default: `0.95`): Lambda for General Advantage Estimation (GAE).
    *   `-minibatch_size` (int, default: `32`): Mini-batch size for each epoch.
    *   `-ent-coef` (float, default: `0.01`): Entropy coefficient for exploration.
    *   `-vf-coef` (float, default: `0.5`): Value function loss coefficient.
    *   `-num-agents` (int, default: `2`): Specifies the number of agents in the custom environment.

*   **Script:** `train_multiagent_ca_ppo.py`

    This script trains multiple PPO agents for a multi-agent reinforcement learning (MARL) task. Each agent has its own policy and value function, for the blade group control and the cooling tower control. It specifically employs a Centralized Action (CA) mechanism. The script is designed to work with MARL environments.

    **Basic Run Command:** `python train_multiagent_ca_ppo.py`

    The key configurable parameters for this script is identical to `train_mh_ma_ca_ppo.py`