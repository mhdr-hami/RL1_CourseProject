from tkinter.tix import Tree
from Environment import Pendulum_Env
import numpy as np
import matplotlib.pyplot as plt
import os

'''
alpha = [0.001,0.01,0.2,0.5] 
etha = [0.001,0.01,0.1]
'''

class Diff_Qlearner():

    def load_run(self):
        runs_name=os.listdir('Check-Runs-Total-Reward')
        final_table=runs_name[-1]
        print(final_table)
        run=np.load('Check-Runs-Total-Reward/'+final_table)
        return run
    
    def load_run_rr(self):
        runs_name=os.listdir('Check-Runs-Reward-Rate')
        final_table=runs_name[-1]
        run=np.load('Check-Runs-Reward-Rate/'+final_table)
        return run


    def total_reward(self,runs):
        avg_Treward_all_runs=np.average(runs,axis=0)
        plt.plot(avg_Treward_all_runs)
        plt.ylabel('Total Rewards - 50 runs')
        plt.xlabel('Steps')
        plt.savefig('plot_1.png')
        plt.close()

    
    def reward_rate_plot(self,runs):
        
        avg_Treward_all_runs=np.average(runs,axis=0)
        plt.plot(avg_Treward_all_runs)
        plt.ylabel('Reward Rate')
        plt.xlabel('Timesteps')
        plt.savefig('plot_2.png')
        plt.close()
    
    def __init__(self,environment,step_num,learning_rate,Avg_R_lr,run_nums,check):
        self.run_num=run_nums
        self.env_p=environment
        self.n_step=step_num
        self.lr=learning_rate
        self.epsilon=1
        self.epsilon_decay=0.9995 #0.9995   ## 0.99975
        self.min_epsilon=0.1
        self.reward_rate=0
        self.lr_avg=Avg_R_lr
        self.check = check
        self.q_table=np.random.uniform(-2, 0,(self.env_p.desired_states_num+[self.env_p.desired_action_num]))

    def run(self):

        all_runs=np.zeros((self.run_num,self.n_step))

        all_runs1=np.zeros((self.run_num,self.n_step))

        aaa = 0
        print("Total : ",aaa)
        
        for run_index in range(self.run_num):

            self.epsilon=1

            pretrained=False
            if pretrained == True:
                self.epsilon=0
                self.q_table=np.load('qtable_4000.npy')
            
            np.random.seed(run_index)
            
            
            total_reward=0
            self.reward_rate=0

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
                all_runs1[run_index][step] = self.reward_rate

                new_state_index=self.env_p.get_discrete_state(new_obs)
                
                if pretrained==False:
                    target_error= reward - self.reward_rate + np.max(self.q_table[new_state_index])-self.q_table[cur_state_index][action_index]
                    self.q_table[cur_state_index][action_index]+=self.lr*(target_error)
                    self.reward_rate+=self.lr_avg*self.lr*(target_error)

                
                cur_state_index=new_state_index

                if step % 1000 ==0 and pretrained == False:
                    if self.epsilon >= self.min_epsilon:
                        self.epsilon*=self.epsilon_decay
                
                if step % 100000 == 0 and pretrained ==False:
                    #print(step," Steps has passed !")
                    #print("Epsilon is : ",self.epsilon)
                    pass
                
                if step % 1000000 == 0 and pretrained == False:
                    np.save(os.path.join('Check-QTable', 'q_'+str(step)+'_run'+str(run_index)), self.q_table)
                    #np.save('q_'+str(step)+'_run'+str(run_index), self.q_table)
            print("Run Number ",run_index)
        
        print("--------------------")
        
        
        self.total_reward(all_runs)
        self.reward_rate_plot(all_runs1)
        np.save(os.path.join('Check-Runs-Total-Reward', 'run_'+str(run_index)+'data'), all_runs)
        np.save(os.path.join('Check-Runs-Reward-Rate', 'run_'+str(run_index)+'data'), all_runs1)
        
        if self.check :
            np.save(os.path.join('Diff_Params', 'run_'+str(run_index)+'_alpha_'+str(self.lr)+'_etha_'+str(self.lr_avg)), all_runs1)


        self.env_p.env.close()


    def action(self,action):
        action=np.array([action])
        obs, reward, done, _,_ = self.env_p.env.step(action)
        return obs,reward,done
    
    

if __name__=="__main__":
    print("kkk")
    sensetivity_analysis=True
    env_name = "Pendulum-v1"
    env_p= Pendulum_Env(env_name)
    agent= Diff_Qlearner(env_p,5000000,0.2,0.01,1,True) ### my best setting 0.2 and 0.01
    if sensetivity_analysis == False:
        agent.run()
    else:
        alpha_range=[0.001,0.01,0.2,0.5,0.9]
        etha_range=[0.001,0.01,0.1,1,2]
        for alpha in alpha_range:
            for etha in etha_range:
                agent= Diff_Qlearner(env_p,5000000,alpha,etha,5,True)
                agent.run()


