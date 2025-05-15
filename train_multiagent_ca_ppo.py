import os
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from multiagent_ca_ppo import multiagent_ppo_dtde
from frontier_env import SmallFrontierModel

def batchify_observations(observation_dict):
    """
    This function takes a dictionary of observations and returns a batch of observations.
    """
    # Initialize an empty list to store the batch of observations
    batch_dict = {'CDUCAB': [], 'CT': []}

    # Iterate through the dictionary and append each observation to the batch
    for key, value in observation_dict.items():
        if key!= 'cooling-tower-1':
            batch_dict['CDUCAB'].append(value)
        else:
            batch_dict['CT'].append(value)

    batch_dict['CDUCAB'] = np.array(batch_dict['CDUCAB'])
    batch_dict['CT'] = np.array(batch_dict['CT'])
    
    # Convert the list to a numpy array
    return batch_dict

def categorize_actions(actions):
    
    actions_categories_dict = {'CDUCAB': ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5'],
                                          'CT': ['cooling-tower-1']}
    categorized_actions = {}
    for agent_id, action in actions.items():
        if agent_id == 'CDUCAB':
            for i, action_category in enumerate(actions_categories_dict[agent_id]):
                categorized_actions[action_category] = action[i]
        else:
            categorized_actions[actions_categories_dict[agent_id][0]] = action
            
    return categorized_actions

# pylint: disable=C0303,C0301,C0116,C0103,C0209,W1514,W0311
################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "SmallFrontierModel"  # environment name

    max_ep_len = 200                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e3)          # save model frequency (in num timesteps)
    best_reward = {'CDUCAB' : float('-inf'),
                   'CT' : float('-inf')}  # initialize best reward as negative infinity
    print_avg_reward = 0                # initialize average reward

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 1      # update policy every n timesteps
    K_epochs = 50               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.80            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 123         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)
    run_num_pretrained = 3    #### change this to prevent overwriting weights in same env_name folder
    reward_fn_version = 2
    exogen_gen_v = 2
    env = SmallFrontierModel(use_reward_shaping = f'reward_shaping_v{reward_fn_version}', exogen_gen_v = exogen_gen_v)
    print("Using Reward Shaping Version : ", env.use_reward_shaping)
    # placeholder agent_mdp_dict    
    agent_mdp_dict = {'CDUCAB' : {'state_dim': env.observation_space['cdu-cabinet-1'].shape[0], 'action_dim': env.action_space['cdu-cabinet-1'].shape[0], 
                                  'num_centralized_actions' : 5,
                                    'lr_actor': lr_actor, 'lr_critic': lr_critic, 'gamma': gamma, 'K_epochs': K_epochs, 'eps_clip': eps_clip, 
                                    'has_continuous_action_space': True, 'action_std_init': action_std},
                      
                     'CT' : {'state_dim': env.observation_space['cooling-tower-1'].shape[0], 'action_dim': env.action_space['cooling-tower-1'].n, 
                                        'num_centralized_actions' : 1,
                                        'lr_actor': lr_actor, 'lr_critic': lr_critic, 'gamma': gamma, 'K_epochs': K_epochs, 'eps_clip': eps_clip, 
                                        'has_continuous_action_space': False, 'action_std_init': action_std}
    }

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "MA_CA_PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/MA_CA_PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    directory = "MA_CA_PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    # create checkpoint_path for each multi agent in a for loop. Assume there is a dictionary with list of agents called agent_mdp_dict
    checkpoint_path = {}
    for agent_id in agent_mdp_dict.keys():
        checkpoint_path[agent_id] = directory + "PPO_{}_{}_{}_agent_{}.pth".format(env_name, random_seed, run_num_pretrained, agent_id)
        print("save checkpoint path : " + checkpoint_path[agent_id])
    #####################################################


    ############# print all hyperparameters #############
    for agent_id, agent_info in agent_mdp_dict.items():
        print("--------------------------------------------------------------------------------------------")
        print("Agent:", agent_id)
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", agent_info['state_dim'])
        print("action space dimension : ", agent_info['action_dim'])
        print("--------------------------------------------------------------------------------------------")
        if agent_info['has_continuous_action_space']:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", agent_info['action_std_init'])
            print("decay rate of std of action distribution : ", action_std_decay_rate)
            print("minimum std of action distribution : ", min_action_std)
            print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", agent_info['K_epochs'])
        print("PPO epsilon clip : ", agent_info['eps_clip'])
        print("discount factor (gamma) : ", agent_info['gamma'])
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", agent_info['lr_actor'])
        print("optimizer learning rate critic : ", agent_info['lr_critic'])
        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)  # pylint: disable=no-member
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = multiagent_ppo_dtde(agent_mdp_dict)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")
    
    ################# tensorboard logging ################
    writer = SummaryWriter()
    
    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward_CDUCAB, reward_CT\n')

    # printing and logging variables
    print_running_reward = {'CDUCAB' : 0, 'CT' : 0}
    print_running_episodes = 0
    print_avg_reward = {'CDUCAB' : 0, 'CT' : 0}

    log_running_reward = {'CDUCAB' : 0, 'CT' : 0}
    log_running_episodes = 0
    log_avg_reward = {'CDUCAB' : 0, 'CT' : 0}
    
    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        batch_state_dict = batchify_observations(state)
        current_ep_reward = {'CDUCAB' : 0, 'CT' : 0}
        # create custom tb logging for cabinet episode reward
        cabinet_totalepisode_reward = {'cdu-cabinet-1':0, 'cdu-cabinet-2':0, 'cdu-cabinet-3':0, 'cdu-cabinet-4':0, 'cdu-cabinet-5':0}

        for _ in range(1, max_ep_len+1):

            actions = {}  # list to store actions for each agent

            # select action for CDU setups
            action = ppo_agent.select_action(batch_state_dict['CDUCAB'], 'CDUCAB')
            actions['CDUCAB'] = action
            # select action for cooling tower
            action = ppo_agent.select_action(batch_state_dict['CT'], 'CT')
            actions['CT'] = action
            
            # categorize actions in to dict format for env
            categorized_actions = categorize_actions(actions)

            state, rewards, done, _ = env.step(categorized_actions)
            batch_state_dict = batchify_observations(state)
            
            # collect next state information for each agent; needed for calculating advantage to track the last state
            # since this is a non terminating environment
            ppo_agent.agents['CDUCAB'].next_state = batch_state_dict['CDUCAB']
            ppo_agent.agents['CT'].next_state = batch_state_dict['CT']
            
            # collect reward and is_terminals for the cducab agent
            ppo_agent.collect_rewards_and_terminals(rewards, done)

            reward = {
                    'CDUCAB' : (rewards['cdu-cabinet-1']+
                                rewards['cdu-cabinet-2']+
                                rewards['cdu-cabinet-3']+
                                rewards['cdu-cabinet-4']+
                                rewards['cdu-cabinet-5'])/5.0, 
                        'CT' : rewards['cooling-tower-1']
                    }
            current_ep_reward['CDUCAB'] += reward['CDUCAB']
            current_ep_reward['CT'] += reward['CT']
            
            for k,v in rewards.items():
                if k!= 'cooling-tower-1':
                    cabinet_totalepisode_reward[k] += v
                
            time_step += 1

            # update PPO agent for each agent
            if time_step % update_timestep == 0:

                CDUCAB_ppo_loss = ppo_agent.update('CDUCAB')
                writer.add_scalar('Loss_CDUCAB', CDUCAB_ppo_loss, time_step)
                CT_ppo_loss = ppo_agent.update('CT')
                writer.add_scalar('Loss_CT', CT_ppo_loss, time_step)

            # if continuous action space; then decay action std of ouput action distribution for each agent
            if time_step % action_std_decay_freq == 0:
                for agent_id in agent_mdp_dict.keys():
                    if agent_mdp_dict[agent_id]['has_continuous_action_space']:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std, agent_id)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode for each agent
                log_avg_reward['CDUCAB'] = log_running_reward['CDUCAB'] / log_running_episodes
                log_avg_reward['CDUCAB'] = round(log_avg_reward['CDUCAB'], 4)
                
                log_avg_reward['CT'] = log_running_reward['CT'] / log_running_episodes
                log_avg_reward['CT'] = round(log_avg_reward['CT'], 4)

                log_f.write('Episode No. : {}, timestep : {}, CDUCAB : {}, CT : {}\n'.format(i_episode, time_step, 
                                                                                             log_avg_reward['CDUCAB'], log_avg_reward['CT']))
                log_f.flush()

                log_running_reward = {'CDUCAB' : 0, 'CT' : 0}
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                print_avg_reward['CDUCAB'] = print_running_reward['CDUCAB'] / print_running_episodes
                print_avg_reward['CDUCAB'] = round(print_avg_reward['CDUCAB'], 2)
                
                print_avg_reward['CT'] = print_running_reward['CT'] / print_running_episodes
                print_avg_reward['CT'] = round(print_avg_reward['CT'], 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward CDUCAB : {} \t\t Average Reward CT: {}".format(i_episode, time_step,
                                                                                                print_avg_reward['CDUCAB'], print_avg_reward['CT'] ))

                print_running_reward = {'CDUCAB' : 0, 'CT' : 0}
                print_running_episodes = 0

            # save model weights for each agent
            if (time_step % save_model_freq  == 0):
                for agent_id in agent_mdp_dict.keys():
                    
                    if (print_avg_reward[agent_id] > best_reward[agent_id]):
                        best_reward[agent_id] = print_avg_reward[agent_id]
                        print("--------------------------------------------------------------------------------------------")
                        print(f"saving model for agent {agent_id} at : " + checkpoint_path[agent_id])
                        ppo_agent.save(checkpoint_path[agent_id], agent_id)
                        print(f"{agent_id} model saved")

                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                
                    

            # collect energy and temperature data from info dictionary
            # energy_trace.append(info['coo.Q_flow'])
            # server1_temp_trace.append(info['serverblock1.heatCapacitor.T'])
            
            # break; if the episode is over
            # if done:
            #     break
            
        
        # Log variables from the info dictionary to TensorBoard
        writer.add_scalar('Episode Reward CDUCAB', current_ep_reward['CDUCAB'], i_episode)
        writer.add_scalar('Episode Reward CT', current_ep_reward['CT'], i_episode) 
        
        # log per cabinet reward
        for k,v in cabinet_totalepisode_reward.items():
            writer.add_scalar(f'Episode Reward {k}', v, i_episode)  
        
        # writer.add_scalar('Energy/CumulativeOneEpisode(kWh)', sum(energy_trace)/1000, i_episode)
        # writer.add_scalar('Server1_Temperature/mean', np.mean(server1_temp_trace) - 273.15, i_episode)
        
        print_running_reward['CDUCAB'] += current_ep_reward['CDUCAB']
        print_running_reward['CT'] += current_ep_reward['CT']
        print_running_episodes += 1

        log_running_reward['CDUCAB'] += current_ep_reward['CDUCAB']
        log_running_reward['CT'] += current_ep_reward['CT']
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at: ", start_time)
    print("Finished training at: ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    