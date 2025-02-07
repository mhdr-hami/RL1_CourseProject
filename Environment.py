from mimetypes import init
import gym

import numpy as np


class Pendulum_Env():
    
    def __init__(self,env_name):    
        
        # best setting
        # self.desired_action_num=17
        # self.desired_states_num=[21,21,65]

        # second setting
        # self.desired_action_num=17
        # self.desired_states_num=[11,11,33]

        # third setting
        self.desired_action_num=17
        self.desired_states_num=[41,41,129]
        
        #self.env=gym.make(env_name,render_mode="human")
        #self.env=gym.make(env_name,render_mode="rgb_array")
        self.env=gym.make(env_name)
        #self.env=gym.make(env_name)
        self.action_space=self.get_discrete_action(self.env.action_space,self.desired_action_num)
        self.state_change_step=self.discrete_state_params(self.env.observation_space,self.desired_states_num)   
        
        #self.state_space=self.get_state_space(self.env.observation_space)

        print(self.action_space)
        print(self.state_change_step)
    
    # def get_state_space(self,obs):
        
    #     for i in range():
    #         action_set[i] = self.env.action_space.low[0] + (i * self.state_change_step[0])
    
    def get_discrete_action(self,env_actions,desired_action_num):
        
        # max_action = env_actions.high
        # min_action = env_actions.low
       
        # action_change_step=(max_action - min_action)/(desired_action_num)
        # actions=np.arange(min_action,max_action,action_change_step)

        action_set={}

        # for index , element in enumerate(actions):
        #     action_set[index]=element
        
        discrete_action_space_win_size = (self.env.action_space.high - self.env.action_space.low) / (self.desired_action_num-1)
        for i in range(self.desired_action_num):
            action_set[i] = self.env.action_space.low[0] + (i * discrete_action_space_win_size[0])
        
        return action_set
    
    def discrete_state_params(self,env_states,desired_states_num):

        max_states_list = env_states.high
        min_states_list = env_states.low
        state_change_step=(max_states_list -min_states_list)/ [k-1 for k in desired_states_num]
        return state_change_step
    
    def get_discrete_state(self,state):
        #return tuple(((state[0]-self.env.observation_space.low)/self.state_change_step).astype(np.int32))
        #return tuple(((state-self.env.observation_space.low)/self.state_change_step).astype(np.int32)),tuple(((state[0]-self.env.observation_space.low)/self.state_change_step).astype(np.int32))
        return tuple(((state-self.env.observation_space.low)/self.state_change_step).astype(np.int32))

# if __name__=="__main__":

    # env_name = "Pendulum-v1"
    # env= Pendulum_Env(env_name)
    # # print(env.env.observation_space)
    # # print("Action space is ==> ",env.env.action_space)
    # # print("State change step is ==> ",env.state_change_step)
    # init_state=env.env.reset()[0]
    # print("Original format of initial state is:",init_state)
    # d_state1,d_state2= env.get_discrete_state(init_state)
    # print("discrete format of init state is ==> ",d_state1)
    # print("discrete format of init state is ==> ",d_state2)
    # d_action_set= env.get_discrete_action(env.env.action_space,env.desired_action_num)
    # print("discrete format of action set is ==>  ",d_action_set)













