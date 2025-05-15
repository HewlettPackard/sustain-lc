<div style="text-align: center;">
    <img src="assets/sustain_lc_logo.png" alt="Logo" width="200"/>
</div>

# SustainLC: Benchmark for End-to-End Control of Liquid Cooling in Sustainable Data Centers

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
