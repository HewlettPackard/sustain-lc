import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Dirichlet

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self,):

        self.actions = {'top-level':[], 'valve-level':[]}
        self.logprobs = {'top-level':[], 'valve-level':[]}
        self.states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions['top-level'][:]
        del self.actions['valve-level'][:]
        del self.logprobs['top-level'][:]
        del self.logprobs['valve-level'][:]
        del self.states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
    def append_batch(self, states, actions, action_logprobs, states_vals):

        for state, top_level_action, top_level_action_logprob, valve_level_action, valve_level_action_logprob, state_val in \
                        zip(states, actions['top-level'], action_logprobs['top-level'], actions['valve-level'], action_logprobs['valve-level'],states_vals):
            
            self.states.append(state)
            self.actions['top-level'].append(top_level_action)
            self.logprobs['top-level'].append(top_level_action_logprob)
            self.actions['valve-level'].append(valve_level_action)
            self.logprobs['valve-level'].append(valve_level_action_logprob)
            self.state_values.append(state_val)


class MultiHeadActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(MultiHeadActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((self.action_dim['top-level'],), action_std_init * action_std_init).to(device)
        
        # actor
        self.backbone = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh()
                    )
        self.top_level_action_features = nn.Linear(64, self.action_dim['top-level'])
        self.top_level_actions = nn.Tanh()
        self.valve_level_features = nn.Linear(64, self.action_dim['valve-level'])
        self.valve_level_actions = nn.Softmax(dim=-1)

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim['top-level'],), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, return_imp_wts = False):
              
        x = self.backbone(state)
        
        # top level actions
        top_level_actions_mean = self.top_level_actions(self.top_level_action_features(x))
        top_level_cov_mat = torch.diag(self.action_var)
        top_level_dist = MultivariateNormal(top_level_actions_mean, top_level_cov_mat)
        top_level_actions = top_level_dist.sample()
        top_level_actions_logprob = top_level_dist.log_prob(top_level_actions)
        
        # valve level actions
        valve_level_features = self.valve_level_features(x)
        valve_concentration = F.softplus(valve_level_features) + 1e-3  # ensures positivity  #pylint: disable=E1102
        valve_level_dist = Dirichlet(valve_concentration)
    
        # valve_level_actions_mean = self.valve_level_actions(valve_level_features)
        # valve_level_action_length = valve_level_actions_mean.shape[1]
        # valve_level_cov_mat = torch.diag(0.001*torch.ones(valve_level_action_length)).to(device)
        # valve_level_dist = MultivariateNormal(valve_level_actions_mean, valve_level_cov_mat)
        valve_level_actions = valve_level_dist.sample()
        valve_level_actions_logprob = valve_level_dist.log_prob(valve_level_actions)
        
        state_val = self.critic(state)
        
        if return_imp_wts:
            # calculate importance weights for the actions
            imp_wts = self.calculate_experience_weight_ls(top_level_actions, top_level_actions_mean, top_level_dist, action_type = 'top-level')
            
            return top_level_actions.detach(), valve_level_actions.detach(), top_level_actions_logprob.detach(), \
                valve_level_actions_logprob.detach(), state_val.detach(), imp_wts.detach()
        else:
             return top_level_actions.detach(), valve_level_actions.detach(), top_level_actions_logprob.detach(), \
                valve_level_actions_logprob.detach(), state_val.detach()
                
    def calculate_experience_weight_ls(self, actions, actions_mean, action_dist, action_type = 'top-level'):
        
        if action_type == 'top-level':
            
            low = torch.ones(self.action_dim['top-level']).to(device) * -1
            low = low.unsqueeze(0).expand_as(actions_mean)
            high = torch.ones(self.action_dim['top-level']).to(device)
            high = high.unsqueeze(0).expand_as(actions_mean)
            dist_to_low = torch.abs(actions_mean - low)
            dist_to_high = torch.abs(actions_mean - high)
            a_boundary = torch.where(dist_to_low > dist_to_high, low, high)
            
            # calculate l(s_t) = log π*(a*_t | s_t) - min_a log π*(a | s_t)
            imp_wts = action_dist.log_prob(actions) - action_dist.log_prob(a_boundary)
            
            return imp_wts
        
        else:
            raise NotImplementedError("Importance weight calculation for valve level actions not implemented.")

    def evaluate(self, state, action):
            
        x = self.backbone(state)
        
        # top level actions
        top_level_action_mean = self.top_level_actions(self.top_level_action_features(x))
        top_level_action_var = self.action_var.expand_as(top_level_action_mean)
        top_level_cov_mat = torch.diag_embed(top_level_action_var).to(device)
        top_level_dist = MultivariateNormal(top_level_action_mean, top_level_cov_mat)
         
        # valve level actions
        valve_level_features = self.valve_level_features(x)
        valve_concentration = F.softplus(valve_level_features) + 1e-3  # ensures positivity  #pylint: disable=E1102
        valve_level_dist = Dirichlet(valve_concentration)
        
        top_level_action_logprobs = top_level_dist.log_prob(action['top-level'])
        valve_level_action_logprobs = valve_level_dist.log_prob(action['valve-level'])
        action_logprobs = top_level_action_logprobs + valve_level_action_logprobs
        dist_entropy = top_level_dist.entropy() + valve_level_dist.entropy()    
        
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class MultiHead_CA_PPO:
    def __init__(self, state_dim, action_dim, num_centralized_actions,
                 lr_actor = 0.0003, lr_critic = 0.001 , gamma = 0.80,
                 K_epochs = 50, eps_clip = 0.2,
                 action_std_init=0.6):
        
        self.num_centralized_actions = num_centralized_actions
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer_dict = {}
        for i in range(self.num_centralized_actions):
            self.buffer_dict[f'action_{i+1}'] = RolloutBuffer()
        

        self.policy = MultiHeadActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': list(self.policy.backbone.parameters()) + 
                                   list(self.policy.top_level_action_features.parameters()) + 
                                   list(self.policy.valve_level_features.parameters()), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = MultiHeadActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.next_state = None

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, states, return_imp_wts = False):

        with torch.no_grad():
            states = torch.FloatTensor(states).to(device)
            
            if return_imp_wts:
                top_level_actions, valve_level_actions, top_level_actions_logprobs, valve_level_actions_logprobs, state_vals, imp_wts = self.policy_old.act(states, return_imp_wts = True)
            else:
                top_level_actions, valve_level_actions, top_level_actions_logprobs, valve_level_actions_logprobs, state_vals = self.policy_old.act(states, return_imp_wts = False)
            
            if self.num_centralized_actions == 1:
                states, state_vals = states.unsqueeze(0), state_vals.unsqueeze(0)
                top_level_actions, valve_level_actions = top_level_actions.unsqueeze(0), valve_level_actions.unsqueeze(0)
                top_level_actions_logprobs, valve_level_actions_logprobs = top_level_actions_logprobs.unsqueeze(0), valve_level_actions_logprobs.unsqueeze(0)
    
            for i, state, top_level_action, top_level_action_logprob, valve_level_action, valve_level_action_logprob, state_val in \
                                                                    zip(range(self.num_centralized_actions), states,
                                                                    top_level_actions, top_level_actions_logprobs,
                                                                    valve_level_actions, valve_level_actions_logprobs,
                                                                    state_vals):
                self.buffer_dict[f'action_{i+1}'].states.append(state)
                self.buffer_dict[f'action_{i+1}'].actions['top-level'].append(top_level_action)
                self.buffer_dict[f'action_{i+1}'].logprobs['top-level'].append(top_level_action_logprob)
                self.buffer_dict[f'action_{i+1}'].actions['valve-level'].append(valve_level_action)
                self.buffer_dict[f'action_{i+1}'].logprobs['valve-level'].append(valve_level_action_logprob)
                self.buffer_dict[f'action_{i+1}'].state_values.append(state_val)
            if return_imp_wts:
                return top_level_actions.detach().cpu().numpy(), valve_level_actions.detach().cpu().numpy(), imp_wts.detach().cpu().numpy()   
            else:
                return top_level_actions.detach().cpu().numpy(), valve_level_actions.detach().cpu().numpy()

    def prepare_update_data(self,):
        
        # torchify and move next_state data to device for update
        next_state = torch.FloatTensor(self.next_state).to(device)
        # calculate next state value
        with torch.no_grad():
            next_state_value = self.policy_old.critic(next_state).detach()
        
        # for each action buffer we now do the following
        mc_returns, discounted_reward, old_states, old_state_values, advantages = {}, {}, {}, {}, {}
        old_top_level_actions, old_top_level_logprobs, old_valve_level_actions, old_valve_level_logprobs = {}, {}, {}, {}
        
        for i in range(self.num_centralized_actions):
            mc_returns[f'action_{i+1}'] = []
            discounted_reward[f'action_{i+1}'] = next_state_value[i]
            
            for reward, is_terminal in zip(reversed(self.buffer_dict[f'action_{i+1}'].rewards), reversed(self.buffer_dict[f'action_{i+1}'].is_terminals)):
                if is_terminal:
                    discounted_reward[f'action_{i+1}'] = 0
                discounted_reward[f'action_{i+1}'] = reward + (self.gamma * discounted_reward[f'action_{i+1}'])
                mc_returns[f'action_{i+1}'].insert(0, discounted_reward[f'action_{i+1}'])
                
            # Normalizing the mc_returns
            mc_returns[f'action_{i+1}'] = torch.tensor(mc_returns[f'action_{i+1}'], dtype=torch.float32).to(device)
            mc_returns[f'action_{i+1}'] = (mc_returns[f'action_{i+1}'] - mc_returns[f'action_{i+1}'].mean()) / (mc_returns[f'action_{i+1}'].std() + 1e-7)
        
            # convert list to tensor
            old_states[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].states, dim=0)).detach().to(device)
            old_top_level_actions[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].actions['top-level'], dim=0)).detach().to(device)
            old_top_level_logprobs[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].logprobs['top-level'], dim=0)).detach().to(device)
            old_valve_level_actions[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].actions['valve-level'], dim=0)).detach().to(device)
            old_valve_level_logprobs[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].logprobs['valve-level'], dim=0)).detach().to(device)
            old_state_values[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].state_values, dim=0)).detach().to(device)
            
            # calculate advantages
            advantages[f'action_{i+1}'] = mc_returns[f'action_{i+1}'].detach() - old_state_values[f'action_{i+1}'].detach()
            
        # now collect all the necessary variables in to their single counterparts by stacking them
        # and moving them to the device
        advantages = torch.cat([advantages[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        mc_returns = torch.cat([mc_returns[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_states = torch.cat([old_states[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_top_level_actions = torch.cat([old_top_level_actions[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_top_level_logprobs = torch.cat([old_top_level_logprobs[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_valve_level_actions = torch.cat([old_valve_level_actions[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_valve_level_logprobs = torch.cat([old_valve_level_logprobs[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_state_values = torch.cat([old_state_values[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        
        # now we have all the data we need to update the policy
        # return the variables to the update method
        return mc_returns, discounted_reward, old_states, old_top_level_actions, old_top_level_logprobs, \
            old_valve_level_actions, old_valve_level_logprobs, old_state_values, advantages
            

    def update(self):
        
        # Monte Carlo estimate of returns
        mc_returns, discounted_reward, old_states, old_top_level_actions, old_top_level_logprobs, \
            old_valve_level_actions, old_valve_level_logprobs, old_state_values, advantages = self.prepare_update_data()      

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Shuffle the data
            indices = torch.randperm(old_states.size(0))
            old_states = old_states[indices]
            old_top_level_actions = old_top_level_actions[indices]
            old_top_level_logprobs = old_top_level_logprobs[indices]
            old_valve_level_actions = old_valve_level_actions[indices]
            old_valve_level_logprobs = old_valve_level_logprobs[indices]
            advantages = advantages[indices]
            mc_returns = mc_returns[indices]

            # Split the data into minibatches
            minibatch_size = 32
            num_minibatches = old_states.size(0) // minibatch_size
            for i in range(num_minibatches):
                start = i * minibatch_size
                end = (i + 1) * minibatch_size

                # Get the minibatch data
                minibatch_old_states = old_states[start:end]
                minibatch_old_top_level_actions = old_top_level_actions[start:end]
                minibatch_old_top_level_logprobs = old_top_level_logprobs[start:end]
                minibatch_old_valve_level_actions = old_valve_level_actions[start:end]
                minibatch_old_valve_level_logprobs = old_valve_level_logprobs[start:end]
                minibatch_advantages = advantages[start:end]
                minibatch_mc_returns = mc_returns[start:end]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(minibatch_old_states, {'top-level': minibatch_old_top_level_actions,
                                                                                                   'valve-level': minibatch_old_valve_level_actions})

                # match state_values tensor dimensions with mc_returns tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - minibatch_old_top_level_logprobs.detach() - minibatch_old_valve_level_logprobs.detach())

                # Finding Surrogate Loss  
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * minibatch_advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, minibatch_mc_returns) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        for i in range(self.num_centralized_actions):
            self.buffer_dict[f'action_{i+1}'].clear()
        
        # return loss for tensorboard logging
        return loss.mean().detach().cpu().numpy()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
