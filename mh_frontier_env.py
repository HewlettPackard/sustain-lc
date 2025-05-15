# this will create a new non gym env class that uses the frontier model env 
# it will be used for decomposing the frontier model action space so that
# we can use a multihead policy to choose the actions and also use a slower simulation rate

import numpy as np

from gymnasium import spaces

from frontier_env import SmallFrontierModel


class MH_SmallFrontierModel:
    
    def __init__(self, *args, **kwargs):
        
        # add a keyword argument called subsample_rate to the kwargs
        kwargs['subsample_rate'] = kwargs.get('subsample_rate', 40)
        kwargs['do_valve_softmax'] = kwargs.get('do_valve_softmax', False)
        
        self.env = SmallFrontierModel(*args, **kwargs)
        
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Dict({
            'cdu-cabinet-1' : spaces.Dict({'top-level':spaces.Box(low=-1,high=1,shape=(2,)),
                               'valve-level':spaces.Box(low=0,high=1,shape=(3,))
                               }),
            'cdu-cabinet-2' : spaces.Dict({'top-level':spaces.Box(low=-1,high=1,shape=(2,)),
                                'valve-level':spaces.Box(low=0,high=1,shape=(3,))
                                 }),
            'cdu-cabinet-3' : spaces.Dict({'top-level':spaces.Box(low=-1,high=1,shape=(2,)),
                                'valve-level':spaces.Box(low=0,high=1,shape=(3,))
                                 }),
            'cdu-cabinet-4' : spaces.Dict({'top-level':spaces.Box(low=-1,high=1,shape=(2,)),
                                'valve-level':spaces.Box(low=0,high=1,shape=(3,))
                                    }),
            'cdu-cabinet-5' : spaces.Dict({'top-level':spaces.Box(low=-1,high=1,shape=(2,)),
                                'valve-level':spaces.Box(low=0,high=1,shape=(3,))
                                    }),
            'cooling-tower-1': spaces.Discrete(len(self.env.cooling_tower_action_decoding)),
        })
        
    def seed(self, seed=None):
        self.env.seed(seed)
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        
        # since internally, smallfrontiermodel assumes actions in range [-1,1] we 
        # have to convert the valve-level actions to the range [-1,1]
        # formula is x_minus1_1 = 2 (x_zero_1 -0)/(1-0) - 1
        
        for cabinet_num in range(5):
            action[f'cdu-cabinet-{cabinet_num+1}']['valve-level'] = 2 * action[f'cdu-cabinet-{cabinet_num+1}']['valve-level'] - 1
        
        # convert the action format to one for smallfrontiermodel
        new_action = {}
        for key,val in action.items():
            if key != 'cooling-tower-1':
                new_action[key] = np.concatenate([val['top-level'],val['valve-level']])
            else:
                new_action[key] = val
        
        return self.env.step(new_action)
    
    def close(self):
        self.env.close()

