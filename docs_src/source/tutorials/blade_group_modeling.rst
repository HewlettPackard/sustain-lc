Blade Group and CDU Modeling
============================

The thermodynamics of each blade-group (BG) with liquid-cooled plates is governed by:

Heat capacitor
--------------

.. math::

   C \,\frac{dT}{dt} \;=\; Q_{\mathrm{port}}(t)

where :math:`T` is the BG temperature and :math:`Q_{\mathrm{port}}` is the net heat flow, here equal to the server’s load:

.. math::

   Q_{\mathrm{port}} = P_{\mathrm{branch}}

Conduction model
----------------

.. math::

   Q_{\mathrm{flow}} \;=\; G_{c}\,(T_{\mathrm{solid}} - T_{\mathrm{fluid}})

with :math:`G_{c}` the convective conductance (function of coolant properties and flow rate :math:`m_{\mathrm{flow}}`).

Overall balance
---------------

.. math::

   \Phi\bigl(Q_{\mathrm{port}}\bigr) + Q_{\mathrm{flow}}
   \;=\; C \,\frac{dT_{\mathrm{server}}}{dt}

Here :math:`\Phi` is a polynomial fit (typically quadratic term 0.015, linear 1) to expose nonlinearity for RL vs. heuristic controllers.  
Server-load trace example: ``input_04-07-24.csv``.
