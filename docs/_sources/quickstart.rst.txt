Quickstart Guide for Sustain-LC
================================

1. **Run the Default Frontier Environment**

   Save the snippet below as :file:`quickstart.py`:

   .. code-block:: python

      from sustainlc.envs.frontier import FrontierEnv

      # Create and reset the environment
      env = FrontierEnv()
      obs = env.reset()
      print(f"Initial observation: {obs}")

      # Simple 5-step control loop
      for step in range(5):
          action = env.action_space.sample()
          obs, reward, done, info = env.step(action)
          print(
              f"Step {step}: action={action}, reward={reward}, "
              f"blade_inlet_temps={info['blade_inlet_temps']}"
          )
          if done:
              break

      env.close()

   Then run:

   .. code-block:: bash

      python quickstart.py

2. **Inspect the Output**

   You should see your initial observation followed by five timesteps printing the sampled action, reward, and blade inlet temperatures.

3. **Next Steps & Further Reading**

   - Advanced modeling with AutoCSM: see :doc:`Advanced AutoCSM Usage <tutorials/advanced_autocsm_usage>`
   - Blade-group and CDU thermodynamics: see :doc:`Blade Group Modeling <tutorials/blade_group_modeling>`
