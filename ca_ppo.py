import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

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
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
    def append_batch(self, states, actions, action_logprobs, states_vals):
        for state, action, action_logprob, state_val in zip(states, actions, action_logprobs, states_vals):
            self.states.append(state)
            self.actions.append(action)
            self.logprobs.append(action_logprob)
            self.state_values.append(state_val)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, return_imp_wts = False):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var)  # .unsqueeze(dim=0)  # Not sure why the original implementation had the unsqueeze; oh for multiple actions arranged in a batch
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        if return_imp_wts & (not self.has_continuous_action_space):
            # collect the logprob of all possible discrete actions from the categorical distribution dist and apply torch min to get the minimum logprob
            min_log_prob = torch.min(dist.log_prob(torch.arange(self.action_dim).to(device)))  # .repeat(action.size(0),1)
            if action.ndim >1:
                min_log_prob = min_log_prob.repeat(action.size(0), 1)
            impt_wts = action_logprob - min_log_prob
            return action.detach(), action_logprob.detach(), state_val.detach(), impt_wts.detach()
        
        elif return_imp_wts & self.has_continuous_action_space:
            raise NotImplementedError("Importance weights not implemented for continuous action space")    
        else:
            return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class CA_PPO:
    def __init__(self, state_dim, action_dim, has_continuous_action_space, num_centralized_actions,
                 lr_actor = 0.0003, lr_critic = 0.001 , gamma = 0.80,
                 K_epochs = 50, eps_clip = 0.2,
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space
        self.num_centralized_actions = num_centralized_actions

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer_dict = {}
        for i in range(self.num_centralized_actions):
            self.buffer_dict[f'action_{i+1}'] = RolloutBuffer()
        

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.next_state = None

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, states, return_imp_wts = False):

        with torch.no_grad():
            states = torch.FloatTensor(states).to(device)
            if return_imp_wts:
                actions, action_logprobs, state_vals, imp_wts = self.policy_old.act(states, return_imp_wts = True)
            else:
                actions, action_logprobs, state_vals = self.policy_old.act(states, return_imp_wts = False)

        if self.num_centralized_actions == 1:
            states, actions, action_logprobs, state_vals = states.unsqueeze(0), actions.unsqueeze(0), action_logprobs.unsqueeze(0), state_vals.unsqueeze(0) 
        
        for i, state, action, action_logprob, state_val in zip(range(self.num_centralized_actions), states, actions, action_logprobs, state_vals):
            self.buffer_dict[f'action_{i+1}'].states.append(state)
            self.buffer_dict[f'action_{i+1}'].actions.append(action)
            self.buffer_dict[f'action_{i+1}'].logprobs.append(action_logprob)
            self.buffer_dict[f'action_{i+1}'].state_values.append(state_val)  
        if self.num_centralized_actions == 1:
            if return_imp_wts:
                return actions.item(), imp_wts.item()
            else:
                return actions.item()
        if return_imp_wts:
            return actions.detach().cpu().numpy(), imp_wts.detach().cpu().numpy()
        else:
            return actions.detach().cpu().numpy(), 
    
    def prepare_update_data(self,):
        
        # torchify and move next_state data to device for update
        next_state = torch.FloatTensor(self.next_state).to(device)
        # calculate next state value
        with torch.no_grad():
            next_state_value = self.policy_old.critic(next_state).detach()
        
        # for each action buffer we now do the following
        mc_returns, discounted_reward, old_states, old_actions, old_logprobs, old_state_values, advantages = {}, {}, {}, {}, {}, {}, {}
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
            old_actions[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].actions, dim=0)).detach().to(device)
            old_logprobs[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].logprobs, dim=0)).detach().to(device)
            old_state_values[f'action_{i+1}'] = torch.squeeze(torch.stack(self.buffer_dict[f'action_{i+1}'].state_values, dim=0)).detach().to(device)
            
            # calculate advantages
            advantages[f'action_{i+1}'] = mc_returns[f'action_{i+1}'].detach() - old_state_values[f'action_{i+1}'].detach()
            
        # now collect all the necessary variables in to their single counterparts by stacking them
        # and moving them to the device
        advantages = torch.cat([advantages[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        mc_returns = torch.cat([mc_returns[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_states = torch.cat([old_states[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_actions = torch.cat([old_actions[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_logprobs = torch.cat([old_logprobs[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        old_state_values = torch.cat([old_state_values[f'action_{i+1}'] for i in range(self.num_centralized_actions)], dim=0).to(device)
        
        # now we have all the data we need to update the policy
        # return the variables to the update method
        return mc_returns, discounted_reward, old_states, old_actions, old_logprobs, old_state_values, advantages
            

    def update(self):
        
        # Monte Carlo estimate of returns
        mc_returns, discounted_reward, old_states, old_actions, old_logprobs, old_state_values, advantages = self.prepare_update_data()      

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Shuffle the data
            indices = torch.randperm(old_states.size(0))
            old_states = old_states[indices]
            old_actions = old_actions[indices]
            old_logprobs = old_logprobs[indices]
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
                minibatch_old_actions = old_actions[start:end]
                minibatch_old_logprobs = old_logprobs[start:end]
                minibatch_advantages = advantages[start:end]
                minibatch_mc_returns = mc_returns[start:end]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(minibatch_old_states, minibatch_old_actions)

                # match state_values tensor dimensions with mc_returns tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - minibatch_old_logprobs.detach())

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
