Cooling Tower Modeling
======================

The cooling tower model simulates the removal of heat from the condenser water loop to ambient air by evaporative cooling, and is governed by the following key relationships.

Mass continuity
----------------

Assuming no bleed or blow-down, the water mass flow through the tower is conserved:

.. math::

   \dot{m}_{w,\mathrm{in}} = \dot{m}_{w,\mathrm{out}}
   = \dot{m}_w

Energy balance
--------------

The sensible heat removed from the water equals the tower heat duty:

.. math::

   \dot{m}_w\,c_{p,w}\,(T_{w,\mathrm{in}} - T_{w,\mathrm{out}})
   = Q_{\mathrm{tower}}

where:

- :math:`\dot{m}_w` is the water mass flow rate  
- :math:`c_{p,w}` is the specific heat capacity of water  
- :math:`T_{w,\mathrm{in}}` and :math:`T_{w,\mathrm{out}}` are the inlet and outlet water temperatures  

Overall heat‐transfer model
---------------------------

Model the tower as a counter-flow heat exchanger with overall conductance :math:`UA` and log-mean temperature difference :math:`\Delta T_{\mathrm{lm}}`:

.. math::

   Q_{\mathrm{tower}} = UA \,\Delta T_{\mathrm{lm}}

with

.. math::

   \Delta T_{\mathrm{lm}}
   = \frac{
       (T_{w,\mathrm{out}} - T_{a,\mathrm{in}})
       \;-\;
       (T_{w,\mathrm{in}}  - T_{a,\mathrm{out}})
     }{
       \ln\!\displaystyle\frac{T_{w,\mathrm{out}} - T_{a,\mathrm{in}}}
                              {T_{w,\mathrm{in}}  - T_{a,\mathrm{out}}}
     }

where:

- :math:`T_{a,\mathrm{in}}` and :math:`T_{a,\mathrm{out}}` are the inlet and outlet air wet-bulb temperatures  

Evaporative mass loss
---------------------

The evaporated water mass flow is tied to the latent heat of vaporization:

.. math::

   \dot{m}_{\mathrm{evap}}
   = \frac{Q_{\mathrm{tower}}}{h_{fg}}

where :math:`h_{fg}` is the latent heat of vaporization of water.

Fan power consumption
---------------------

Air is drawn through the tower by a fan against a pressure rise :math:`\Delta p_a`. The electrical power required is:

.. math::

   P_{\mathrm{fan}}
   = \frac{\Delta p_a \,\dot V_a}{\eta_{\mathrm{fan}}}

where:

- :math:`\dot V_a` is the volumetric air flow rate  
- :math:`\eta_{\mathrm{fan}}` is the fan efficiency  

Usage in Sustain-LC
-------------------

The above equations are implemented in the cooling-tower submodel of the Sustain-LC Gymnasium environment. Users can adjust parameters such as :math:`UA`, :math:`\dot{m}_w`, and fan characteristics to study the impacts on thermal regulation and energy consumption.  
