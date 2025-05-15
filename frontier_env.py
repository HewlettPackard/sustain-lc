import os  # pylint: disable=missing-module-docstring
import copy
import numpy as np
import pyfmi  # pylint: disable=import-error
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from scipy.special import softmax

# suppress trailing whitespace warnings using pylint 
# pylint: disable=C0303, C0301, C0103, W0223

FMU_PATH = os.path.dirname(os.path.abspath(__file__))+"/LC_Frontier_5Cabinet_4_17_25.fmu"
EXOGENOUS_VAR_PATH = 'input_04-07-24.csv'
# self.model = pyfmi.load_fmu(FMU_PATH, kind='CS')

class exogenous_variable_generator:
    
    # read the csv in to an iterator
    def __init__(self, path, Towb_offset_in_K = 10, nCDUs = 5, nBranches = 3, parallel_nCabinets = 5, subsample_rate = 1):
        self.exogenous_var = pd.read_csv(path)
        # Option 1:  Cap values beyond 2 standard deviations in the 2nd to 6th columns
        for col in self.exogenous_var.columns[1:1+nCDUs]:
            mean = self.exogenous_var[col].mean()
            std = self.exogenous_var[col].std()
            upper_limit = mean + 0.1 * std
            lower_limit = mean - 1.75 * std
            self.exogenous_var[col] = self.exogenous_var[col].clip(lower=lower_limit, upper=upper_limit)
        self.exogenous_var.iloc[:,-1] += 273.15 + Towb_offset_in_K  # convert the last column to Kelvin and adding 10K to prevent power saturation in CT
        
        # # Option 2: Repopulate with values sampled from a gaussian cenetered on mean and std 
        # mean = self.exogenous_var.iloc[:,1:1+n].mean()
        # std = self.exogenous_var.iloc[:,1:1+n].std()
        # for col in self.exogenous_var.columns[1:1+n]:
        #     self.exogenous_var[col] = np.random.normal(loc=mean[col], scale=std[col], size=len(self.exogenous_var)).clip(min=50000.0,max=100000)
        # self.exogenous_var.iloc[:,-1] += 273.15 + Towb_offset_in_K  # convert the last column to Kelvin and adding 10K to prevent power saturation in CT 
        
        # Option 3: Shrink around mean or any quantile
        # percentage = 0.10  #   # Change this value to the desired percentage
        # mean = self.exogenous_var.quantile(0.15)
        # scaled_df = (self.exogenous_var - mean) * percentage + mean
        # self.exogenous_var.iloc[:,1:1+n] = scaled_df.iloc[:,1:1+n]
        # self.exogenous_var.iloc[:,-1] += 273.15 + Towb_offset_in_K  # convert the last column to Kelvin and adding 10K to prevent power saturation in CT 
        
        # convert the data frame to a numpy ndarray
        self.exogenous_var = self.exogenous_var.to_numpy()
        
        Q_flow_totals = self.exogenous_var[:,1:1+nCDUs]/parallel_nCabinets  # divide by nCabinets since we have parallel arrangement???
        Q_flow_totals /= nBranches  # since power input is divied between nBranches in modified cabinet model
        Q_flow_totals = Q_flow_totals.repeat(nBranches, axis=1).round(2)
        # do numpy roll on the required column by required elements; 300 elements is 1 hr
        columns_to_roll_dict = {1:1800, 2:3600, 4:1800, 5:3600, 7:1800, 8:3600, 10:1800, 11:3600, 13:1800, 14:3600}
        for col, roll in columns_to_roll_dict.items():
            Q_flow_totals[:,col] = np.roll(Q_flow_totals[:,col], roll, axis=0)
        self.exogenous_var_final = np.concatenate([Q_flow_totals, self.exogenous_var[:,-1].reshape(-1,1)], axis=1)
        
        # currently the numpy rows indictate data at 15 second intervals
        # we want to subsample the data to 10 minute intervals
        self.exogenous_var_final = self.exogenous_var_final[::subsample_rate]
        
        
    def iterate_cyclically(self,):
        while True:
            for row in self.exogenous_var_final:
                yield row
                
class exogenous_variable_generator_2:
    
    # read the csv in to an iterator
    def __init__(self, path, Towb_offset_in_K = 10, nCDUs = 5, nBranches = 3, parallel_nCabinets = 5,
                 smoothing_kernel_size = 50, subsample_rate = 1, hru_e_ntu = 0.90, use_hru = False):
        total_num_cabinets = 25
        self.exogenous_var = pd.read_csv(path)
        # Option 1:  Cap values beyond 2 standard deviations in the 2nd to total_num_cabinets-th columns
        for col in self.exogenous_var.columns[1:1+total_num_cabinets]:
            mean = self.exogenous_var[col].mean()
            std = self.exogenous_var[col].std()
            upper_limit = mean + 0.1 * std
            lower_limit = mean - 1.75 * std
            self.exogenous_var[col] = self.exogenous_var[col].clip(lower=lower_limit, upper=upper_limit)
        self.exogenous_var.iloc[:,-1] += 273.15 + Towb_offset_in_K  # convert the last column to Kelvin and adding 10K to prevent power saturation in CT
        
        # convert the data frame to a numpy ndarray
        self.exogenous_var = self.exogenous_var.to_numpy()
        
        Q_flow_totals = self.exogenous_var[:,1:1+total_num_cabinets]/parallel_nCabinets
        Q_flow_totals /= nBranches  # since power input is divied between nBranches in modified cabinet model
        Q_flow_totals = Q_flow_totals.repeat(nBranches, axis=1).round(2)
        # do numpy roll on the required column by required elements; 300 elements is 1 hr
        columns_to_roll_dict = {}
        for i in range(1, total_num_cabinets*nBranches):
            if i % 3 != 0:
                columns_to_roll_dict[i] = 1800 * (i % nBranches)
        for col, roll in columns_to_roll_dict.items():
            Q_flow_totals[:,col] = np.roll(Q_flow_totals[:,col], roll, axis=0)
            
        # since we can only sample up to 5 cabinets for this fmu, we consider 15(5 cabinets and 3 columns per cabinate) 
        # columns at a time in Q_flow_totals 
        # and stack them one on top of the other to get 5 cabinets
        Q_flow_totals = np.concatenate([Q_flow_totals[:,i:i+15] for i in range(0, total_num_cabinets*nBranches, 15)], axis=0)
        
        
        kernel_size = smoothing_kernel_size
        kernel = np.ones(kernel_size) / kernel_size
        # softmax to stretch the values with 3 columns at a time
        for i in range(0, Q_flow_totals.shape[1],3):
            max_value = np.max(Q_flow_totals[:,i:i+3])
            Q_flow_totals[:,i:i+3] = softmax(Q_flow_totals[:,i:i+3], axis=1) * max_value
            # smoothen the columns of data
            Q_flow_totals[:,i] = np.convolve(Q_flow_totals[:,i], kernel, mode='same')
            Q_flow_totals[:,i+1] = np.convolve(Q_flow_totals[:,i+1], kernel, mode='same')
            Q_flow_totals[:,i+2] = np.convolve(Q_flow_totals[:,i+2], kernel, mode='same')
            
        if use_hru:
            # assume Heat Resuse is using up hru_e_ntu = 0.90 of energy from the system
            Q_flow_totals = Q_flow_totals * hru_e_ntu
        
        # the last column of self.exogenous_var has to be repeated now to match the number of rows in Q_flow_totals
        towb = self.exogenous_var[:,-1].repeat(5).reshape(-1,1)
        
        self.exogenous_var_final = np.concatenate([Q_flow_totals, towb], axis=1)
        
        # currently the numpy rows indictate data at 15 second intervals
        # we want to subsample the data to 10 minute intervals
        self.exogenous_var_final = self.exogenous_var_final[::subsample_rate]
        
    def iterate_cyclically(self,):
        while True:
            for row in self.exogenous_var_final:
                yield row 

class SmallFrontierModel(gym.Env):
    
    CT_nominal_power_per_cell = 0.55*149140  # watts; taken from the cooling tower model v2 made in Dymola
    Towb_offset_in_K = 15.0
    variable_ranges = {
        
        'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt' : [298.15, 313.15],  # K
        
        'cabinet_temperature_K' : [273.15, 373.15],  # K
        'valve_flow_rate':[0.0,12.0],  # m^3/s
        
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].CabRet_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].CabRet_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].CabRet_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].CabRet_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].CabRet_pT.T' : [273.15, 373.15],  # K
        
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].CabSup_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].CabSup_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].CabSup_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].CabSup_pT.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].CabSup_pT.T' : [273.15, 373.15],  # K
        
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].valveCDU.m_flow' : [0.0, 12.0],  # m^3/s
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].valveCDU.m_flow' : [0.0, 12.0],  # m^3/s
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].valveCDU.m_flow' : [0.0, 12.0],  # m^3/s
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].valveCDU.m_flow' : [0.0, 12.0],  # m^3/s
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].valveCDU.m_flow' : [0.0, 12.0],  # m^3/s
        
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].CT.PFan' : [0.0, CT_nominal_power_per_cell],  # watts
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[2].CT.PFan' : [0.0, CT_nominal_power_per_cell],  # watts
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[3].CT.PFan' : [0.0, CT_nominal_power_per_cell],  # watts
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[4].CT.PFan' : [0.0, CT_nominal_power_per_cell],  # watts
        
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_1.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_1.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_1.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_1.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_1.port.T' : [273.15, 373.15],  # K
        
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_2.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_2.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_2.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_2.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_2.port.T' : [273.15, 373.15],  # K
        
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_3.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_3.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_3.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_3.port.T' : [273.15, 373.15],  # K
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_3.port.T' : [273.15, 373.15],  # K
        
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].controls.p_CTWR_Setpoint_Model.T_CT_setpoint' : [280.15, 305.15],  # K
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].waterSPTLvg' : [273.15,373.15],  # K
        
        # exogenous inputs
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Q_flow_total' : [0.0, 1e6],  # J/s
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Q_flow_total' : [0.0, 1e6],  # J/s
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Q_flow_total' : [0.0, 1e6],  # J/s
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Q_flow_total' : [0.0, 1e6],  # J/s
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Q_flow_total' : [0.0, 1e6],  # J/s
        
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3':[0.0, 0.34e6], # J/s
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3':[0.0, 0.34e6], # J/s
        
        
        'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb' : [270.15, 373.15],  # K
        
        # action inputs
        'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_Tsec_supply_nom_RL' : [20.0, 40.0],  # C
        'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_Tsec_supply_nom_RL' : [20.0, 40.0],  # C
        'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_Tsec_supply_nom_RL' : [20.0, 40.0],  # C
        'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_Tsec_supply_nom_RL' : [20.0, 40.0],  # C
        'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_Tsec_supply_nom_RL' : [20.0, 40.0],  # C
        
        
        'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_dp_nom_RL' : [25.0, 38.0], # 
        'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_dp_nom_RL' : [25.0, 38.0], # 
        'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_dp_nom_RL' : [25.0, 38.0], # 
        'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_dp_nom_RL' : [25.0, 38.0], # 
        'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_dp_nom_RL' : [25.0, 38.0], # 
        
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[1]' : [0.0, 1.0],  # unitless
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[2]' : [0.0, 1.0],  # unitless
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[3]' : [0.0, 1.0],  # unitless
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[1]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[2]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[3]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[1]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[2]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[3]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[1]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[2]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[3]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[1]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[2]' : [0.0, 1.0], # unitless
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[3]' : [0.0, 1.0], # unitless
    }
    
    def __init__(self, start_time=0, stop_time=24*60*60, step_size=15.0,use_reward_shaping='reward_shaping_v0',
                 exogen_gen_v = 1, subsample_rate = 1, do_valve_softmax = True):
        
        try:
            self.fmu = pyfmi.load_fmu(FMU_PATH, kind='CS',log_level=0)
            print(f"FMU file loaded correctly: {FMU_PATH}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error loading FMU file: {e}")
            
        self.start_time = start_time
        self.stop_time = stop_time
         
        # Initialize the FMU
        self.fmu.setup_experiment(start_time=self.start_time, stop_time=self.stop_time)
        self.step_size = step_size
        self.fmu.initialize()
        self.current_time = 0
        
        ################ Modify code block here to change the action and observation space ################
        self.observation_vars = {'cdu-cabinet-1' : ['simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_1.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_2.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_3.port.T',
                                                    'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1',
                                                    'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2',
                                                    'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3'],
                                 
                                 'cdu-cabinet-2' : ['simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_1.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_2.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_3.port.T',
                                                    'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1',
                                                    'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2',
                                                    'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3'],
                                 
                                 'cdu-cabinet-3' : ['simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_1.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_2.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_3.port.T',
                                                    'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1',
                                                    'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2',
                                                    'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3'],
                                  
                                'cdu-cabinet-4' : ['simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_1.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_2.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_3.port.T',
                                                    'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1',
                                                    'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2',
                                                    'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3'],
                                
                                'cdu-cabinet-5' : ['simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_1.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_2.port.T',
                                                    'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_3.port.T',
                                                    'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1',
                                                    'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2',
                                                    'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3'],
                                  
                                'cooling-tower-1': ['simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].CT.PFan',
                                                    'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[2].CT.PFan',
                                                    'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].waterSPTLvg',
                                                    'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb']
                                }
        
        self.observation_space = spaces.Dict({
            'cdu-cabinet-1': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-2': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-3': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-4': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-5': spaces.Box(low=-1, high=1, shape=(6,)),
            'cooling-tower-1': spaces.Box(low=-1, high=1, shape=(4,)),
        })
        
        self.raw_observation_space_max = {'cdu-cabinet-1': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-1']]),
                                          'cdu-cabinet-2': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-2']]),
                                          'cdu-cabinet-3': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-3']]),
                                          'cdu-cabinet-4': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-4']]),
                                          'cdu-cabinet-5': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-5']]),
                                         'cooling-tower-1':np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cooling-tower-1']])
                                         }
        self.raw_observation_space_min = {'cdu-cabinet-1': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-1']]),
                                          'cdu-cabinet-2': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-2']]),
                                          'cdu-cabinet-3': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-3']]),
                                          'cdu-cabinet-4': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-4']]),
                                          'cdu-cabinet-5': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-5']]),
                                          'cooling-tower-1':np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cooling-tower-1']])
                                         }

        self.action_vars = {
            'cdu-cabinet-1': [
                                'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_Tsec_supply_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_dp_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[1]',
                                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[2]',
                                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[3]'
                                ],
            'cdu-cabinet-2': [
                                'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_Tsec_supply_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_dp_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[1]',
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[2]',
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[3]'
                                ],
            'cdu-cabinet-3': [
                                'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_Tsec_supply_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_dp_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[1]',
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[2]',
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[3]'
                                ],
            'cdu-cabinet-4': [
                                'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_Tsec_supply_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_dp_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[1]',
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[2]',
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[3]'
                                ],
            'cdu-cabinet-5': [
                                'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_Tsec_supply_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_dp_nom_RL',
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[1]',
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[2]',
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[3]'
                                ],
            'cooling-tower-1': [
                                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt'
                                ]
        }
        
        # self.cooling_tower_action_decoding = {0:-1.0,
        #                                       1:-0.75,
        #                                       2:-0.5,
        #                                       3:-0.25,
        #                                       4:0,
        #                                       5:0.25,
        #                                       6:0.5,
        #                                       7:0.75,
        #                                       8:1.0}
        self.cooling_tower_action_decoding = {0:-0.20,
                                              1:-0.15,
                                              2:-0.10,
                                              3:-0.05,
                                              4:0,
                                              5:0.05,
                                              6:0.10,
                                              7:0.15,
                                              8:0.20}
        
        self.action_space = spaces.Dict({
            'cdu-cabinet-1': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-2': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-3': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-4': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-5': spaces.Box(low=-1, high=1, shape=(5,)),
            'cooling-tower-1': spaces.Discrete(len(self.cooling_tower_action_decoding)),
        })
        
        
        self.raw_action_space_max = {'cdu-cabinet-1': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-1']]),
                                     'cdu-cabinet-2': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-2']]),
                                     'cdu-cabinet-3': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-3']]),
                                     'cdu-cabinet-4': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-4']]),
                                     'cdu-cabinet-5': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-5']]),
                                     'cooling-tower-1': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cooling-tower-1']])
                                    }
        self.raw_action_space_min = {'cdu-cabinet-1': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-1']]),
                                    'cdu-cabinet-2': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-2']]),
                                    'cdu-cabinet-3': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-3']]),
                                    'cdu-cabinet-4': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-4']]),
                                    'cdu-cabinet-5': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-5']]),
                                    'cooling-tower-1': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cooling-tower-1']])
                                    }
            
        # select the indices for collecting exogenous data, combining this with action and obs space code block
        # so that they can be changed in case observation space changes
        self.nCDUs = 5
        self.parallel_nCabinets = 3
        self.nBranches = 3
        # Create an instance of the exogenous variable class
        self.exogenous_var = [  'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1',
                                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2',
                                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3',
                                
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1',
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2',
                                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3',
                                
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1',
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2',
                                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3',
                                
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1',
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2',
                                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3',
                                
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1',
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2',
                                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3',
                                
                                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb']
        
        if exogen_gen_v == 1:
            self.iter_exogenous_var = exogenous_variable_generator(EXOGENOUS_VAR_PATH, Towb_offset_in_K = self.Towb_offset_in_K, subsample_rate = subsample_rate).iterate_cyclically()
        elif exogen_gen_v == 2:
            self.iter_exogenous_var = exogenous_variable_generator_2(EXOGENOUS_VAR_PATH, Towb_offset_in_K = self.Towb_offset_in_K, subsample_rate = subsample_rate).iterate_cyclically()
        else:
            raise ValueError("Invalid version of exogenous variable generator. Please use 1 or 2.")

        #######################################################################################################
        
        self.done = {}
        self.reward = {}
        self.info = {}
        self.scaled_action = None
        self.previous_state = None
        self.use_reward_shaping = use_reward_shaping
        # valve softmaxing
        self.do_valve_softmax = do_valve_softmax
        
    def seed(self, seed=None):  # pylint: disable=arguments-differ
        pass
    
    def action_inverse_mapper(self,action):  # it is doing from [-1, 1] to raw action range mapping
        unscaled_action = {'cdu-cabinet-1': None, 
                           'cdu-cabinet-2': None,
                           'cdu-cabinet-3': None,
                           'cdu-cabinet-4': None,
                           'cdu-cabinet-5': None,
                           'cooling-tower-1': None}
        assert type(action) == dict, "Action must be a dict"
        for key,val in action.items():
            if key == 'cooling-tower-1':
                unscaled_action[key] = self.cooling_tower_action_decoding[val]
            else:
                assert type(val) == np.ndarray, "Actions for each category must be a numpy np.ndarray before being sent to the environment"
                unscaled_action[key] = self.raw_action_space_min[key] + 0.5*(val + 1)*(self.raw_action_space_max[key] - self.raw_action_space_min[key])
        return unscaled_action
    
    def observation_mapper(self,raw_observation):  # it is doing -1 to 1 mapping
        scaled_observation = {'cdu-cabinet-1': None,
                              'cdu-cabinet-2': None,
                              'cdu-cabinet-3': None,
                              'cdu-cabinet-4': None,
                              'cdu-cabinet-5': None,
                              'cooling-tower-1': None}
        assert type(raw_observation) == dict, "Observation must be a dict"
        for key,val in raw_observation.items():
            assert type(val) == np.ndarray, "Observations for each category must be a numpy np.ndarray before being sent from the environment"
            scaled_observation[key] = 2*(val - self.raw_observation_space_min[key])/(self.raw_observation_space_max[key] - self.raw_observation_space_min[key]) - 1
        return scaled_observation
    
    def get_exogenous_var(self):
        # next_state = next(self.iter_exogenous_var)
        
        # Q_flow_totals = next_state[1:1+self.nCDUs]/self.parallel_nCabinets  # divide by nCabinets since we have parallel arrangement???
        # Q_flow_totals /= self.nBranches  # since power input is divied between nBranches in modified cabinet model TODO: Implement non uniform branching of heat
        # Q_flow_totals = list(Q_flow_totals.repeat(self.nBranches).flatten())
        # T_owb = next_state[-1]
        
        return next(self.iter_exogenous_var)
    
    def reset(self,):
        self.fmu.reset()
        self.fmu.setup_experiment(start_time=self.start_time, stop_time=self.stop_time)
        self.fmu.initialize()
        self.current_time = 0
        
        raw_observations = {'cdu-cabinet-1': None,
                            'cdu-cabinet-2': None,
                            'cdu-cabinet-3': None,
                            'cdu-cabinet-4': None,
                            'cdu-cabinet-5': None,
                            'cooling-tower-1': None}
        for key,var_list in self.observation_vars.items():
            raw_observations[key] = np.array([i[0] for i in self.fmu.get(var_list)])
            
        # self.latest_ct_setpoint = 293.15  # K
        self.previous_state = self.observation_mapper(raw_observations)
            
        return self.previous_state
        
    def step(self, action):  # action is of the form dict(str, np.ndarray)
        self.scaled_action = copy.deepcopy(action)
        action = self.action_inverse_mapper(action)
        
        # 0-1 scale the valve stpts for each cabinet; it works by 0-1 scaling every 3 values
        if self.do_valve_softmax:
            for key, val in action.items():
                if key != 'cooling-tower-1':
                    action[key][2:2+3] = np.exp(val[2:2+3]) / np.sum(np.exp(val[2:2+3]))  # pylint: disable=E1137,E1136 
            
        # set the action variables on the fmu in a loop based on the action space
        for key, var_list in self.action_vars.items():
            if key != 'cooling-tower-1':
                self.fmu.set(var_list, [round(i,2) for i in list(action[key])])
            else:
                # new_setpoint = self.latest_ct_setpoint + action[key]
                # clip the value of new_setpoint which is a scalar between "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt" max and min values
                # new_setpoint = np.clip(new_setpoint, self.variable_ranges[var_list[0]][0], self.variable_ranges[var_list[0]][1])
                
                latest_wetbulb = self.fmu.get('simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb')[0]
                rule_based_new_setpoint = latest_wetbulb + 10.0*5/9  # 17.0
                rl_new_setpoint = action[key] + rule_based_new_setpoint
                
                # for the cooling tower, we need to set the action as a single value
                self.fmu.set(var_list, [round(rl_new_setpoint,2)])
        
        # Perform a simulation step
        self.fmu.do_step(current_t=self.current_time, step_size=self.step_size)
        
        # get the exogenous variables;   # where should this be called? Here! so that agent knows next state of exogenous variables
        exogenous_variables = self.get_exogenous_var()
        # set them on the sytem before the next step
        self.fmu.set(self.exogenous_var, list(exogenous_variables))
        
        # collect the required observations space only (this must be after the above set process so that agent knows current state of the fmu)
        raw_observations = {'cdu-cabinet-1': None,
                            'cdu-cabinet-2': None,
                            'cdu-cabinet-3': None,
                            'cdu-cabinet-4': None,
                            'cdu-cabinet-5': None,
                            'cooling-tower-1': None}
        for key,var_list in self.observation_vars.items():
            raw_observations[key] = np.array([i[0] for i in self.fmu.get(var_list)])  # since fmu get returns each variable as a list/tuple
        
        self.info = raw_observations.copy()
        self.info['actions'] = action
        self.info['actions_newsetpoint'] = rl_new_setpoint
        
        # scaled observations
        scaled_observation = self.observation_mapper(raw_observations)
        
        # reward
        self.reward = {}
        self.done = {}
        # choose the reward shaping method for cabinet models (ct models can be added later)
        if self.use_reward_shaping == 'reward_shaping_v0':
            self.reward_shaping_v0(scaled_observation)
        elif self.use_reward_shaping == 'reward_shaping_v1':
            self.reward_shaping_v1()
        elif self.use_reward_shaping == 'reward_shaping_v2':
            self.reward_shaping_v2(scaled_observation)
        else:
            raise ValueError(f"Unknown reward shaping method: {self.use_reward_shaping}")
            
        self.reward['cooling-tower-1'] = (2-scaled_observation['cooling-tower-1'][0:2].sum())/2  # pylint: disable=E1136
        self.done['cooling-tower-1'] = False  # there is no episode termination condition since it is a continuous task
        
        # this HAS to be AFTER reward calculation for the reward_shaping_v1 implementation. Can move elsewhere for other implementations
        self.previous_state = scaled_observation
 
        return scaled_observation, self.reward, self.done, self.info
    
    def reward_shaping_v0(self,scaled_observation):
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            self.reward[key] = (3-scaled_observation[key][0:3].sum())/3  # pylint: disable=E1136
            self.done[key] = False  # there is no episode termination condition since it is a continuous task
        
    def reward_shaping_v1(self,):
        # reward shaping logic here
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            # max abs difference in per-branch-power and per-branch-valve-action across each cabinet is 2.0
            self.reward[key] = 6.0 -abs(self.scaled_action[key][2] - self.previous_state[key][3]) \
            - abs(self.scaled_action[key][3] - self.previous_state[key][4]) - abs(self.scaled_action[key][4] - self.previous_state[key][5])
            self.done[key] = False  # there is no episode termination condition since it is a continuous task
            
    def reward_shaping_v2(self,scaled_observation):
        # combine rewards v0 and v1 per cabinet
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            
            # component 1
            # max abs difference in per-branch-power and per-branch-valve-action across each cabinet is 2.0
            reward_component_1 = 6.0 -abs(self.scaled_action[key][2] - self.previous_state[key][3]) \
            - abs(self.scaled_action[key][3] - self.previous_state[key][4]) - abs(self.scaled_action[key][4] - self.previous_state[key][5])
            # component 2
            reward_component_2 = (3-scaled_observation[key][0:3].sum())/3
            # combine the two components
            w1,w2 = 1.0,3.0
            self.reward[key] = w1*reward_component_1 + w2*reward_component_2
            self.done[key] = False
        
if __name__ == "__main__":
    
    exogen = exogenous_variable_generator_2('input_04-07-24.csv')