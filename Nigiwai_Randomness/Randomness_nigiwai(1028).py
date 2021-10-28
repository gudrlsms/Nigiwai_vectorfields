import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from matplotlib import cm
from scipy.integrate import dblquad
import sympy
import numpy as np
from tqdm import tqdm
import random

# Vadere data
def load_vadere(df, scale, start_frame, end_frame, frame_skip):
    nPed = int(df['pedestrianId'].max())
    df['pedestrianId'] -= 1 # PID start with 0
    X = np.full((nPed,end_frame),np.nan)  # (pid,frame),  x == nan means the person is not in the field
    Y = np.full((nPed,end_frame),np.nan)  # (pid,frame)
    print("Interpolating pedestrian trajectory...")
    for pid,tid,st,et,sx,ex,sy,ey in tqdm(zip(df['pedestrianId'],df['targetId-PID2'],df['simTime'],df['endTime-PID1'],df['startX-PID1'],df['endX-PID1'],df['startY-PID1'],df['endY-PID1']),total=df.shape[0]):
        pid = int(pid)
        start_fr = max(int(round(st/frame_skip)),start_frame)
        end_fr = min(int(round(et/frame_skip))+1,end_frame)
        X[pid,start_fr:end_fr] = np.linspace(sx,ex, end_fr-start_fr)
        Y[pid,start_fr:end_fr] = np.linspace(sy,ey, end_fr-start_fr)
    return(scale*X,scale*Y)

# Vadere pamameters
scale = 1
start_frame = 0
end_frame = 501
frame_skip = 1

# ---------------------- data -----------------------
# Speed difference data
"""
s_fast1000 = pd.read_csv('data/500fr_speed/shop_fast1000.txt',header=0,delim_whitespace=True,dtype='f8')
s_normal1000 = pd.read_csv('data/500fr_speed/shop_normal1000.txt',header=0,delim_whitespace=True,dtype='f8')
s_normal500_fast500 = pd.read_csv('data/500fr_speed/shop_normal500_fast500.txt',header=0,delim_whitespace=True,dtype='f8')
p_fast1000 = pd.read_csv('data/500fr_speed/pass_fast1000.txt',header=0,delim_whitespace=True,dtype='f8')
p_normal1000 = pd.read_csv('data/500fr_speed/pass_normal1000.txt',header=0,delim_whitespace=True,dtype='f8')
p_normal500_fast500 = pd.read_csv('data/500fr_speed/pass_normal500_fast500.txt',header=0,delim_whitespace=True,dtype='f8')

df_list=[p_fast1000, p_normal1000, p_normal500_fast500, s_fast1000, s_normal1000, s_normal500_fast500]
"""

# shopping ratio test data
df_list = []
for i in range(0,11):
    df_list.append(pd.read_csv('data/500fr_ratio_test/{}%.txt'.format(i*10),header=0,delim_whitespace=True,dtype='f8'))

# Consgestion data
"""
s900 = pd.read_csv('data/300fr_spawn/spawn_3.txt',header=0,delim_whitespace=True,dtype='f8')
s1800 = pd.read_csv('data/300fr_spawn/spawn_6.txt',header=0,delim_whitespace=True,dtype='f8')

df_list=[s900, s1800]
"""
# ---------------------- nigiwai indicator -----------------------
# parameters for indicators
fr_number = 9 # number of velocity data for calculation of randomness of each ped
const_each = 0.5 # constant for weight between R_1 and R_2 in indicator for randomness of each pedestrian
const_all = 0 # constant for weight between R_1 and R_2 in indicator for randomness of all pedestrian
const_p = 0.5 # constant for weight between "each" and "all" in nigiwai indicator

indicator_list_graph = []
# Data process
for df in df_list:
    df_vadere = load_vadere(df, scale, start_frame, end_frame, frame_skip)
    
    # Adding Gaussian noise
    noise_std = 0
    for i in range(0,end_frame-1):
        noise = np.random.normal(0,noise_std,len(df_vadere[0]))
        df_vadere[0][:,i] += noise
        noise = np.random.normal(0,noise_std,len(df_vadere[0]))
        df_vadere[1][:,i] += noise
    
    # position coordinate data
    df_x = df_vadere[0]
    df_y = df_vadere[1]
    
    # Delete singular points on simulation data(entrance, exit)
    for i in range(len(df_y)):
        for j in range(len(df_y[i])):
            if df_y[i][j] > 45 or df_y[i][j] < 5:
                df_x[i][j] = np.nan
                df_y[i][j] = np.nan
    
    # velocity data
    df_u = np.full((len(df_x),end_frame),np.nan)
    df_v = np.full((len(df_y),end_frame),np.nan)
    for i in range(end_frame):
        if i == end_frame-1:
            df_u[:,i] = np.full((len(df_x)),np.nan)
            df_v[:,i] = np.full((len(df_x)),np.nan)
        else:
            df_u[:,i] = df_x[:,i+1] - df_x[:,i]
            df_v[:,i] = df_y[:,i+1] - df_y[:,i]
    
    # speed data
    df_norm = np.sqrt(df_u**2 + df_v**2)
    
    # acceleration data
    df_u_acc = np.full((len(df_u),end_frame-1),np.nan)
    df_v_acc = np.full((len(df_v),end_frame-1),np.nan)
    for i in range(end_frame-1):
        if i == end_frame-2:
            df_u_acc[:,i] = np.full((len(df_u)),np.nan)
            df_v_acc[:,i] = np.full((len(df_v)),np.nan)
        else:
            df_u_acc[:,i] = df_u[:,i+1] - df_u[:,i]
            df_v_acc[:,i] = df_v[:,i+1] - df_v[:,i]
    
    # accelerating force data
    df_acc_force = np.sqrt(df_u_acc**2 + df_v_acc**2)
    
    # If velocity is equal to 0(or less than std of noise), it takes former velocity
    for i in range(len(df_norm)):
        for j in range(len(df_norm[i])):
            if df_norm[i][j] <= noise_std:
                if not i==0 and not j==0:
                        df_u[i][j] = df_u[i][j-1]
                        df_v[i][j] = df_v[i][j-1]
    df_norm = np.sqrt(df_u**2 + df_v**2)
    
    # normalize velocity
    for i in range(len(df_u)):
        for j in range(len(df_u[i])):
            if not df_norm[i][j] == 0:
                df_u[i][j] = df_u[i][j] / df_norm[i][j]
                df_v[i][j] = df_v[i][j] / df_norm[i][j]
    
    # normalized velocity data for 2nd trigonometic moment
    df_u2 = df_u**2 - df_v**2
    df_v2 = 2*df_u*df_v
    
    # nigiwai indicator
    indicator_list = []
    for target_fr in range(fr_number+10, round(end_frame/frame_skip)-1):
        # data for "randomness of each ped(function "f")" at target frame range
        df_u_f = df_u[:,target_fr-fr_number:target_fr+1]
        df_v_f = df_v[:,target_fr-fr_number:target_fr+1]
        
        # 2nd moment
        df_u2_f = df_u2[:,target_fr-fr_number:target_fr+1]
        df_v2_f = df_v2[:,target_fr-fr_number:target_fr+1]
    
        # data for function "randomness of all ped(function "g")" at target frame
        df_u_g = df_u[:,target_fr]
        df_v_g = df_v[:,target_fr]
        
        df_u2_g = df_u2[:,target_fr]
        df_v2_g = df_v2[:,target_fr]
        
        df_u_g = df_u_g[~np.isnan(df_u_g)]
        df_v_g = df_v_g[~np.isnan(df_v_g)]
        df_u2_g = df_u2_g[~np.isnan(df_u2_g)]
        df_v2_g = df_v2_g[~np.isnan(df_v2_g)]
        
        # indicator: randomness of each pedestrian(function "f")
        indicator_f_r1 = np.sqrt(np.sum(df_u_f,axis=1)**2 + np.sum(df_v_f,axis=1)**2) / (fr_number+1)
        indicator_f_r2 = np.sqrt(np.sum(df_u2_f,axis=1)**2 + np.sum(df_v2_f,axis=1)**2) / (fr_number+1)
        
        indicator_f = 1 - (const_each*indicator_f_r1 + (1-const_each)*indicator_f_r2)
        
        # indicator: randomness of all pedestrian(function "g")
        if len(df_u_g) == 0:
            indicator_g = 0
        else:
            indicator_g_r1 = np.sqrt(np.sum(df_u_g)**2 + np.sum(df_v_g)**2) / len(df_u_g)
            indicator_g_r2 = np.sqrt(np.sum(df_u2_g)**2 + np.sum(df_v2_g)**2) / len(df_u2_g)
            
            indicator_g = 1 - (const_all*indicator_g_r1 + (1-const_all)*indicator_g_r2)
                
        # indicator: acceleration force
        df_acc_force_target = df_acc_force[:,target_fr-fr_number:target_fr+1]
        indicator_acc_force = np.sum(df_acc_force_target,axis=1) / (fr_number+1)
        
        # Weithed acceleration force
        total_acc_force = np.sum(indicator_acc_force[~np.isnan(indicator_acc_force)])
        
        # indicator: acceleration
        df_u_acc_target = df_u_acc[:,target_fr-fr_number:target_fr+1]
        df_v_acc_target = df_v_acc[:,target_fr-fr_number:target_fr+1]
        indicator_acc = np.sqrt((np.sum(df_u_acc,axis=1))**2 + (np.sum(df_v_acc,axis=1))**2) / (fr_number+1)
        
        # Calculate indicator for each ped        
        indicator_each_ped = indicator_acc_force * indicator_f
        #indicator_each_ped = indicator_f
        #indicator_each_ped = indicator_acc_force
        
        indicator_each_ped = indicator_each_ped[~np.isnan(indicator_each_ped)]
        
        if len(indicator_each_ped) == 0:
            indicator_each_ped = 0
        else:
            indicator_each_ped = np.sum(indicator_each_ped)/len(indicator_each_ped)
            #indicator_each_ped = np.sum(indicator_each_ped)/total_acc_force
            
        # Calculate indicator
        indicator = (indicator_g**const_p) * (indicator_each_ped**(1-const_p))
        #indicator = indicator_each_ped
        #indicator = indicator_g
        
        indicator_list.append(indicator) # generate list of indicators
    
    # moving average for indicator graph
    const_ma = 0.05
    indicator_list_ma = []
    indicator_list_ma.append(indicator_list[0])
    for i in range(len(indicator_list)-1):
        indicator_list_ma.append(const_ma*indicator_list[i+1] + (1-const_ma)*indicator_list_ma[i])
    
    # generate graph
    indicator_list_graph.append(indicator_list_ma)

plt.figure(figsize=(10, 10))
plt.title('Indicator')
x_values = list(range(fr_number+10, round(end_frame/frame_skip)-1))
for i in range(len(indicator_list_graph)):
    plt.plot(x_values, indicator_list_graph[i])
#plt.axis([fr_number+5, round(end_frame/frame_skip)-1, 0.1, 0.6])
plt.legend(['0%','10%', '20%','30%','40%','50%','60%','70%','80%','90%','100%'])
plt.show()