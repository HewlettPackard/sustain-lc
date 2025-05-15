from ca_ppo import CA_PPO
from multihead_ca_ppo import MultiHead_CA_PPO

class multiagent_ppo_dtde:
    
    # DTDE stands for decentralized training with decentralized execution
    
    def __init__(self, agent_mdp_dict, agent_type = 'CA_PPO'):
        
        self.agents = {}
        
        if agent_type == 'CA_PPO':
            for agent_id, agent_mdp in agent_mdp_dict.items():
                self.agents[agent_id] = CA_PPO(**agent_mdp)

        elif agent_type == 'MultiHead_CA_PPO':
            for agent_id, agent_mdp in agent_mdp_dict.items():
                if agent_id == 'CDUCAB':
                    self.agents[agent_id] = MultiHead_CA_PPO(**agent_mdp)
                else:
                    self.agents[agent_id] = CA_PPO(**agent_mdp)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    def select_action(self, state, agent_id, return_imp_wts = False):
        # if agent_id == 'CDUCAB':
        #     return self.agents[agent_id].select_action(state, return_imp_wts = return_imp_wts)
        # else:
        #     return self.agents[agent_id].select_action(state)
        return self.agents[agent_id].select_action(state, return_imp_wts = return_imp_wts)
    
    def decay_action_std(self, action_std_decay_rate, min_action_std, agent_id): 
        self.agents[agent_id].decay_action_std(action_std_decay_rate, min_action_std)
            
    def update(self, agent_id):
        return self.agents[agent_id].update()
    
    def save(self, checkpoint_path, agent_id):
        self.agents[agent_id].save(checkpoint_path)
            
    def load(self, checkpoint_path, agent_id):
        self.agents[agent_id].load(checkpoint_path)
     
    def collect_rewards_and_terminals(self, rewards, terminals):
        for action_id in range(self.agents['CDUCAB'].num_centralized_actions):
            self.agents['CDUCAB'].buffer_dict[f'action_{action_id+1}'].rewards.append(rewards[f'cdu-cabinet-{action_id+1}'])
            self.agents['CDUCAB'].buffer_dict[f'action_{action_id+1}'].is_terminals.append(terminals[f'cdu-cabinet-{action_id+1}'])
        for action_id in range(self.agents['CT'].num_centralized_actions):
            self.agents['CT'].buffer_dict[f'action_{action_id+1}'].rewards.append(rewards[f'cooling-tower-{action_id+1}'])
            self.agents['CT'].buffer_dict[f'action_{action_id+1}'].is_terminals.append(terminals[f'cooling-tower-{action_id+1}'])