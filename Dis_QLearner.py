from Environment import Pendulum_Env
import numpy as np
import matplotlib.pyplot as plt
import os

class Dis_Qlearner():

    def load_run(self,param_check,param_values):
        
        if param_check == False:
            runs_name=os.listdir('Check-Runs-Total-Reward-1')
            final_table=runs_name[-1] ## wrong since os.listdir() gives in random order
            print("The Final Run name is ",final_table)
            run=np.load('Check-Runs-Total-Reward-1/'+final_table)
            return run
        else:
            run_list=[]
            if param_check=="gamma":
                for val in param_values:
                    runs_name=os.listdir(param_check+"_runs/gamma_"+str(val)+'/Check-Runs-Total-Reward-1')
                    final_table=runs_name[-1]
                    print("The Final Run name is ",final_table)
                    run=np.load(param_check+"_runs/gamma_"+str(val)+'/Check-Runs-Total-Reward-1/'+ final_table)
                    run_list.append(run)
                
                return run_list

            elif param_check == 'alpha':
                for val in param_values:
                    runs_name=os.listdir(param_check+"_runs2/alpha_"+str(val)+'/Check-Runs-Total-Reward-1')
                    final_table=runs_name[-1]
                    print("The Final Run name is ",final_table)
                    run=np.load(param_check+"_runs2/alpha_"+str(val)+'/Check-Runs-Total-Reward-1/'+ final_table)
                    run_list.append(run)
                
                return run_list

    
    def total_reward(self,runs):
        avg_Treward_all_runs=np.average(runs,axis=0)
        plt.plot(avg_Treward_all_runs)
        plt.ylabel('Total Rewards')
        plt.xlabel('Steps')
        plt.savefig('plot1.png')

    
    def __init__(self,environment,step_num,learning_rate,gamma,run_nums,param_check):
        
        self.run_num=run_nums
        self.env_p=environment
        
        self.n_step=step_num
        self.lr=learning_rate
        
        self.gamma=gamma
        self.epsilon=1
        
        self.epsilon_decay=0.99975  ### 0.99975  ### default is 25 million steps with 0.9999//// 0.99975 for 10 million
        #### /// 0.9995 for 5 million steps with this setting the diff is the weakest ==> maybe we need to tune diff params // 0.9975 for 1 million steps
        self.min_epsilon=0.1
        
        self.check_option=param_check ### 
        
        if self.check_option:
            self.select_param="alpha"
        
        self.q_table=np.random.uniform(-2, 0,(self.env_p.desired_states_num+[self.env_p.desired_action_num]))
    

    def single_param_tunning(self,active_param,values):
        self.param_values=values
        if active_param == 'gamma':
            
            # if not os.path.exists('gamma_runs'):
            #     os.makedirs('gamma_runs')
            
            for gamma_val in values:
                
                self.gamma=gamma_val
                self.run()
        elif active_param == 'alpha':
            for aplha in values:
                self.lr = aplha
                self.run()

    def run(self):

        all_runs=np.zeros((self.run_num,self.n_step))
                
        for run_index in range(self.run_num):

            self.epsilon=1

            pretrained=False
            if pretrained == True:
                self.epsilon=0
                self.q_table=np.load('qtable_4000.npy')
            
            np.random.seed(run_index)
            
            
            total_reward=0
        
            cur_obs= self.env_p.env.reset()[0]
            cur_state_index=self.env_p.get_discrete_state(cur_obs)
                
            for step in range(self.n_step):
                
                if self.epsilon > np.random.uniform(0,1):
                    action_index = np.random.randint(0, self.env_p.desired_action_num)
                else:
                    action_index = np.argmax(self.q_table[cur_state_index])

                action = self.env_p.action_space[action_index]
                new_obs,reward,_=self.action(action)
                
                total_reward+=reward
                
                all_runs[run_index][step] = total_reward

                new_state_index=self.env_p.get_discrete_state(new_obs)
                
                if pretrained==False:
                    self.q_table[cur_state_index][action_index]+=self.lr*(reward + self.gamma*np.max(self.q_table[new_state_index])-self.q_table[cur_state_index][action_index])

                
                cur_state_index=new_state_index

                if step % 1000 ==0 and pretrained == False:
                    if self.epsilon >= self.min_epsilon:
                        self.epsilon*=self.epsilon_decay
                if step % 100000 == 0 and pretrained ==False:
                    pass
                    # print(step," Steps has passed !")
                    # print("Epsilon is : ",self.epsilon)
                if step % 1000000 == 0 and pretrained == False:
                    #np.save('q_'+str(step)+'_run'+str(run_index), self.q_table)
                    if self.check_option ==False:
                        np.save(os.path.join('Check-QTable-1', 'q_'+str(step)+'_run'+str(run_index)), self.q_table)
                    else:
                        np.save(os.path.join(self.select_param+'_runs2/alpha_'+str(self.lr)+'/Check-QTable-1', 'q_'+str(step)+'_run'+str(run_index)), self.q_table)

        if self.check_option == False:
            self.total_reward(all_runs)
        
        if self.check_option == False:
            np.save(os.path.join('Check-Runs-Total-Reward-1', 'run_'+str(run_index)+'data'), all_runs)
        else:
            np.save(os.path.join(self.select_param+'_runs2/alpha_'+str(self.lr)+'/Check-Runs-Total-Reward-1', 'run_'+str(run_index)+'data'), all_runs)
        
        self.env_p.env.close()


    def action(self,action):
        action=np.array([action])
        obs, reward, done, _,_ = self.env_p.env.step(action)
        return obs,reward,done
    
    

if __name__=="__main__":

    env_name = "Pendulum-v1"
    env_p= Pendulum_Env(env_name)
    agent= Dis_Qlearner(env_p,10000000,0.1,0.9,5,True)
    
    if agent.check_option==True:
        agent.single_param_tunning('alpha',[0.001,0.01,0.2,0.5,0.9])
    else:
        agent.run()
