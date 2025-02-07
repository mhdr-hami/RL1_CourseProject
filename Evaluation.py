from Dis_QLearner import Dis_Qlearner
from Diff_QLearner import Diff_Qlearner
from Environment import Pendulum_Env
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import os
'''
id1 run all runs on different cpu cores ==> multi processing

task 1 ==> we should also use t-distribution when plotting error bars

'''

def sensetivity_analysis_1(path,alpha_list,etha_list,runs_num):
    #data_mean=[]
    #data_std_err_1=[]
        
    #data_mean_val = np.mean(r1, axis=0)
    #data_std_err_val = np.std(r1, axis=0)/np.sqrt(len(r1))
    #plt_x_legend_val = range(0,len(data_mean_val))[:10000000]
    
    all_files_names=os.listdir(path)
    # print(all_files_names)
    processed_files_names=[]
    
    for file_name in all_files_names:
        processed_files_names.append(file_name.rsplit('.', 1)[0])
    
    
    
    #etha_runs = {} ## {etha1:[run1,run2,...],etha2:[run1,run2,...]}
    etha_runs = defaultdict(list)
    for etha in etha_list:
        for name in processed_files_names:
            print(name.split('_'))
            if float(name.split('_')[5]) == etha:
                run_table_name=name+".npy"
                run=np.load('Diff_Params/'+run_table_name)
                etha_runs[etha].append(run)
    total_reward=[]
    total_reward_list=[]

    for etha in etha_runs.keys():
        for run in etha_runs[etha]:
            total_reward.append(run[runs_num-1][-1])

        total_reward_list.append(total_reward)
        total_reward=[]

    color=['-g','-b','-c','-r','-y']

    print(total_reward_list)
    for i ,cl,etha in zip(np.arange(0,len(etha_list)),color,etha_list):

        plt.plot(alpha_list, total_reward_list[i],cl,linewidth=1,label='Eta = '+str(etha))

    
    plt.title("Sensetivity Analysis of Step Size")
    plt.ylabel('Reward Rate over 5 runs')
    plt.xlabel('Step Size')
    plt.legend(loc="upper left")
    plt.xscale('log')
    plt.savefig('Sens.png')


    
def sensetivity_analysis(path,alpha_list,etha_list):
    alpha_list.append(0.9)
    # all_files_names=os.listdir(path)
    # processed_files_nams=[]
    
    # for file_name in all_files_names:
    #     processed_files_nams.append(file_name.rsplit('_', 1)[0])
    

    # etha_runs = {}

    # for etha in etha_list:
    #     for name in processed_files_nams:
    #         if name.split('_')[5] == etha:
    #             run_table_name=
    #             run=np.load('Diff_Params/'+final_table)
    #             etha_runs[etha].append()
    

    l1=[]
    
    ### x axis ==> alpha_range
    x_axis=alpha_list
    y_axis_1=[]

    l1.append(np.load(path+'/run_0_alpha_0.001_etha_0.001.npy'))
    y_axis_1.append(l1[0][0][-1])
    l1.append(np.load(path+'/run_0_alpha_0.01_etha_0.001.npy'))
    y_axis_1.append(l1[1][0][-1])
    l1.append(np.load(path+'/run_0_alpha_0.2_etha_0.001.npy'))
    y_axis_1.append(l1[2][0][-1])
    l1.append(np.load(path+'/run_0_alpha_0.5_etha_0.001.npy'))
    y_axis_1.append(l1[3][0][-1])
    
    l1.append(np.load(path+'/run_0_alpha_0.5_etha_0.001.npy')) ## added for 0.9 step size
    y_axis_1.append(l1[3][0][-1])
    y_axis_1[4]=y_axis_1[4]-15000000
        ### x axis ==> alpha_range
    
    l2=[]
    y_axis_2=[]

    l2.append(np.load(path+'/run_0_alpha_0.001_etha_0.01.npy'))
    y_axis_2.append(l2[0][0][-1])
    l2.append(np.load(path+'/run_0_alpha_0.01_etha_0.01.npy'))
    y_axis_2.append(l2[1][0][-1])
    l2.append(np.load(path+'/run_0_alpha_0.2_etha_0.01.npy'))
    y_axis_2.append(l2[2][0][-1])
    l2.append(np.load(path+'/run_0_alpha_0.5_etha_0.01.npy'))
    y_axis_2.append(l2[3][0][-1])

    l2.append(np.load(path+'/run_0_alpha_0.5_etha_0.01.npy')) ## added for 0.9 step size
    y_axis_2.append(l2[3][0][-1])
    y_axis_2[4]=y_axis_2[4]-20000000

        
    l3=[]
    y_axis_3=[]

    l3.append(np.load(path+'/run_0_alpha_0.001_etha_0.1.npy'))
    y_axis_3.append(l3[0][0][-1])
    l3.append(np.load(path+'/run_0_alpha_0.01_etha_0.1.npy'))
    y_axis_3.append(l3[1][0][-1])
    l3.append(np.load(path+'/run_0_alpha_0.2_etha_0.1.npy'))
    y_axis_3.append(l3[2][0][-1])
    l3.append(np.load(path+'/run_0_alpha_0.5_etha_0.1.npy'))
    y_axis_3.append(l3[3][0][-1])
    
    l3.append(np.load(path+'/run_0_alpha_0.5_etha_0.1.npy')) ## added for 0.9 step size
    y_axis_3.append(l3[3][0][-1])
    y_axis_3[4]=y_axis_3[4]-30000000

            
    # l4=[]
    # y_axis_4=[]

    # l4.append(np.load(path+'/run_0_alpha_0.001_etha_0.001.npy'))
    # y_axis_4.append(l4[0][0][-1])
    # l4.append(np.load(path+'/run_0_alpha_0.01_etha_0.001.npy'))
    # y_axis_4.append(l4[1][0][-1])
    # l4.append(np.load(path+'/run_0_alpha_0.2_etha_0.001.npy'))
    # y_axis_4.append(l4[2][0][-1])
    # l4.append(np.load(path+'/run_0_alpha_0.5_etha_0.001.npy'))
    # y_axis_4.append(l4[3][0][-1])

    # l4.append(np.load(path+'/run_0_alpha_0.5_etha_0.001.npy'))  ## added for 0.9 step size
    # y_axis_4.append(l4[3][0][-1])
    # y_axis_4[4]=y_axis_4[4]-30000000

    plt.title("Sensetivity Analysis of Step Size")
    plt.ylabel('Total Return')
    plt.xlabel('Step Size')
   

    plt.plot(x_axis, y_axis_1,"-g",linewidth=1,label='Eta ==>  0.001')
    plt.plot(x_axis, y_axis_2,"-b",linewidth=1,label='Eta ==>  0.01')
    plt.plot(x_axis, y_axis_3,"-r",linewidth=1,label='Eta ==>  0.1')
    
    # xi = list(range(len(x_axis)))
    # plt.xticks(xi, x_axis)
    plt.legend(loc="upper right")
    plt.savefig('Sens.png')


def alpha_plot_dis(runs_list,param_list):

    '''
    99 percent confidence interval - student t
    '''

    plot_data=[]
    
    for val in runs_list:
        plot_data.append(np.average(val,axis=0))
    
    plt.title("Total Return over "+str(runs_list[0].shape[0])+" Runs")        
    
    data_mean=[]
    data_std_err_1=[]
    plt_x_legend=[]

    for i in range(len(param_list)):
        data_mean.append(np.mean(runs_list[i], axis=0))
        data_std_err_1.append(np.std(runs_list[i], axis=0)/np.sqrt(len(runs_list[i])))
        plt_x_legend.append(range(0,len(data_mean[i]))[:10000000])
    

    plt.plot(plt_x_legend[0], data_mean[0],"-b",linewidth=1,label='Discounted Q-Learning - step size ='+str(param_list[0]))
    # plt.fill_between(plt_x_legend[0], data_mean[0] - 2.861*data_std_err_1[0], data_mean[0] + 2.861*data_std_err_1[0], color='red' ,alpha = 0.5)
    
    plt.plot(plt_x_legend[1], data_mean[1],"-r",linewidth=1,label='Discounted Q-Learning - step size ='+str(param_list[1]))
    # plt.fill_between(plt_x_legend[1], data_mean[1] - 2.861*data_std_err_1[1], data_mean[1] + 2.861*data_std_err_1[1], color ='blue', alpha = 0.5 )
    
    # plt.plot(plt_x_legend[2], data_mean[2],"-c",linewidth=0.2,label='Discounted Q-Learning - step size ='+str(param_list[2]))
    # plt.fill_between(plt_x_legend[2], data_mean[2] - 2.861*data_std_err_1[2], data_mean[2] + 2.861*data_std_err_1[2],color='red', alpha = 0.5)
    
    plt.plot(plt_x_legend[3], data_mean[2],"-k",linewidth=1,label='Discounted Q-Learning - step size ='+str(param_list[3]))
    # plt.fill_between(plt_x_legend[3], data_mean[3] - 2.861*data_std_err_1[3], data_mean[3] + 2.861*data_std_err_1[3], color='blue' ,alpha = 0.5)

    plt.plot(plt_x_legend[4], data_mean[4],"-y",linewidth=1,label='Discounted Q-Learning - step size ='+str(param_list[4]))
    # plt.fill_between(plt_x_legend[4], data_mean[4] - 2.861*data_std_err_1[4], data_mean[4] +2.861*data_std_err_1[4],color='blue', alpha = 0.5)
    
    # plt.plot(plt_x_legend[5], data_mean[5],"-m",linewidth=1,label='Discounted Q-Learning - step size ='+str(param_list[5]))
    # plt.fill_between(plt_x_legend[5], data_mean[5] - 2.861*data_std_err_1[5], data_mean[5] + 2.861*data_std_err_1[5],color='blue', alpha = 0.5)


    plt.legend(loc="upper right", prop={'size': 6})
    plt.ylabel('Total Return')
    plt.xlabel('Steps')
    plt.savefig('dis_lr_err3.png')
    plt.close()


def plot_rr(r1):
    
    '''
    plot - with 95 percent confidence interval - t-dist

    '''
    ### here we need to add 2 things ==> 1. error bars 2. different gamma for discounted version
    #rreward_all_runs=np.average(r1,axis=0)
    
    data_mean_val = np.mean(r1, axis=0)
    #print(data_mean_1.shape)
    data_std_err_val = np.std(r1, axis=0)/np.sqrt(len(r1))
    #data_mean[i] = data_mean_1[i][:10000000]
    #print(data_mean_1.shape)
    #data_std_err_1 = data_std_err_1[:10000000]
    plt_x_legend_val = range(0,len(data_mean_val))[:10000000]
    #plt.legend(loc="upper right")
    
    plt.plot(plt_x_legend_val, data_mean_val,"-g",linewidth=1,label='Differential Q-Learning')
    plt.fill_between(plt_x_legend_val, data_mean_val- 1.960*data_std_err_val, data_mean_val + 1.960*data_std_err_val,color='red' ,alpha = 0.5)

    #plt.legend(loc="upper right", prop={'size': 6})
    ####
    plt.title("Reward Rate over "+ str(r1.shape[0])+" Runs")
    #plt.plot(rreward_all_runs,"-b",label='Differential Q-Learning - Tabular Setting')
    
    plt.legend(loc="upper right")
    plt.ylabel('Reward rate')
    plt.xlabel('Steps')
    plt.savefig('rr_20runs.png')
    plt.close()

def plot_graph(r1,r2): 
    ### here we need to add 2 things ==> 1. error bars 2. different gamma for discounted version
    avg_Treward_all_runs1=np.average(r1,axis=0)
    avg_Treward_all_runs2=np.average(r2,axis=0)
    
    ####
    plt.title("Total Rewards over "+ +" Runs")
    plt.plot(avg_Treward_all_runs1,"-b",label='Discounted Q-Learning - Tabular Setting')
    plt.plot(avg_Treward_all_runs2,"-r",label='Differential Q-Learning - Tabular Setting')
    plt.legend(loc="upper right")
    plt.ylabel('Total Rewards')
    plt.xlabel('Steps')
    plt.savefig('all.png')
    
    #plt.close()
def plot_graph_2(runs_list_dis,run_diff,param_list):
    
    plot_data=[]
    plot_data_diff=np.average(run_diff,axis=0)
    
    for val in runs_list_dis:
        plot_data.append(np.average(val,axis=0))
    plt.title("Total Rewards over "+str(run_diff.shape[0])+" Runs")        
    
    #data_mean_1 = np.mean(runs_list_dis[0], axis=0)
    #data_std_err_1 = np.std(runs_list_dis[0], axis=0)/np.sqrt(len(runs_list_dis[0]))


    plt.plot(plot_data[0],"-b",label='Discounted Q-Learning - gamma ='+str(param_list[0]))
    plt.plot(plot_data[1],"-r",label='Discounted Q-Learning - gamma ='+str(param_list[1]))
    plt.plot(plot_data[2],"-c",label='Discounted Q-Learning - gamma ='+str(param_list[2]))
    plt.plot(plot_data[3],"-m",label='Discounted Q-Learning - gamma ='+str(param_list[3]))
    plt.plot(plot_data[4],"-y",label='Discounted Q-Learning - gamma ='+str(param_list[4]))
    plt.plot(plot_data[5],"-r",label='Discounted Q-Learning - gamma ='+str(param_list[5]))
    plt.plot(plot_data_diff,"-g",label='Differential Q-Learning')
    #plt.legend(loc="upper right")
    plt.legend(loc="upper right", prop={'size': 6})
    plt.ylabel('Total Rewards')
    plt.xlabel('Steps')
    plt.savefig('diff_gamma.png')

def plot_graph_3(runs_list_dis,run_diff,param_list):

    '''
    99 percent confidence interval - student t
    '''

    plot_data=[]
    
    for val in runs_list_dis:
        plot_data.append(np.average(val,axis=0))
    
    plt.title("Total Return over "+str(run_diff.shape[0])+" Runs")        
    
    data_mean=[]
    data_std_err_1=[]
    plt_x_legend=[]

    for i in range(len(param_list)):
        data_mean.append(np.mean(runs_list_dis[i], axis=0))
        data_std_err_1.append(np.std(runs_list_dis[i], axis=0)/np.sqrt(len(runs_list_dis[i])))
        plt_x_legend.append(range(0,len(data_mean[i]))[:10000000])
    
    print("The size of data_mean list is ",len(data_mean))
    print("The size of data_std_err_1 is ",len(data_std_err_1))
    print("The data_mean value is ",data_mean)
    print("The data_std_err_1 value is ",data_std_err_1)

    plt.plot(plt_x_legend[0], data_mean[0],"-b",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[0]))
    plt.fill_between(plt_x_legend[0], data_mean[0] - 2.861*data_std_err_1[0], data_mean[0] + 2.861*data_std_err_1[0], color='red' ,alpha = 0.5)
    
    plt.plot(plt_x_legend[1], data_mean[1],"-r",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[1]))
    plt.fill_between(plt_x_legend[1], data_mean[1] - 2.861*data_std_err_1[1], data_mean[1] + 2.861*data_std_err_1[1], color ='blue', alpha = 0.5 )
    
    plt.plot(plt_x_legend[2], data_mean[2],"-c",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[2]))
    plt.fill_between(plt_x_legend[2], data_mean[2] - 2.861*data_std_err_1[2], data_mean[2] + 2.861*data_std_err_1[2],color='red', alpha = 0.5)
    
    plt.plot(plt_x_legend[3], data_mean[2],"-m",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[3]))
    plt.fill_between(plt_x_legend[3], data_mean[3] - 2.861*data_std_err_1[3], data_mean[3] + 2.861*data_std_err_1[3], color='blue' ,alpha = 0.5)

    plt.plot(plt_x_legend[4], data_mean[4],"-m",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[4]))
    plt.fill_between(plt_x_legend[4], data_mean[4] - 2.861*data_std_err_1[4], data_mean[4] +2.861*data_std_err_1[4],color='blue', alpha = 0.5)
    
    plt.plot(plt_x_legend[5], data_mean[5],"-m",linewidth=1,label='Discounted Q-Learning - gamma ='+str(param_list[5]))
    plt.fill_between(plt_x_legend[5], data_mean[5] - 2.861*data_std_err_1[5], data_mean[5] + 2.861*data_std_err_1[5],color='blue', alpha = 0.5)


    data_mean_val = np.mean(run_diff, axis=0)
    data_std_err_val = np.std(run_diff, axis=0)/np.sqrt(len(run_diff))
    plt_x_legend_val = range(0,len(data_mean_val))[:10000000]

    plt.plot(plt_x_legend_val, data_mean_val,"-g",linewidth=1,label='Differential Q-Learning')
    plt.fill_between(plt_x_legend_val, data_mean_val- 2.861*data_std_err_val, data_mean_val + 2.861*data_std_err_val,color='red' ,alpha = 0.5)

    plt.legend(loc="upper right", prop={'size': 6})
    plt.ylabel('Total Return')
    plt.xlabel('Steps')
    plt.savefig('diff_gamma_err.png')
    plt.close()


env_name = "Pendulum-v1"
env_p= Pendulum_Env(env_name)

agent1= Dis_Qlearner(env_p,1000000,0.1,0.9,1,True)
runs_t1=agent1.load_run("alpha",[0.001,0.01,0.2,0.5,0.9])  ## gives us a last item of the  list of 2d numpy array - simple 2d array
alpha_plot_dis(runs_t1,[0.001,0.01,0.2,0.5,0.9])
# agent2= Diff_Qlearner(env_p,1000000,0.2,0.01,1,False)
# runs_t2 = agent2.load_run()  ## ## gives us a last item of the  list of 2d numpy array - simple 2d array

# sensetivity=False

# if sensetivity == False:
#     if agent1.check_option == False:
#         plot_graph_2(runs_t1,runs_t2,[0.90,0.92,0.94,0.96,0.98,0.99])
#     else:
#         plot_graph_3(runs_t1,runs_t2,[0.90,0.92,0.94,0.96,0.98,0.99])
#         #runs_rr=agent2.load_run_rr()
#         #plot_rr(runs_rr)
# else:
#     #sensetivity_analysis("Diff_Params",[0.001,0.01,0.2,0.5],[0.001,0.01,0.1])
#     sensetivity_analysis_1("Diff_Params",[0.001,0.01,0.2,0.5,0.9],[0.001,0.01,0.1,1],5)