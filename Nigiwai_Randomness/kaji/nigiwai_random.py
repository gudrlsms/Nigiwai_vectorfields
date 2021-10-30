import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from tqdm import tqdm

# Loading vadere data
def load_vadere(df, scale, start_frame, end_frame, frame_skip):
    nPed = int(df['pedestrianId'].max())
    df['pedestrianId'] -= 1 # PID start with 0
    X = np.full((nPed,end_frame),np.nan)  # (pid,frame),  x == nan means the person is not in the field
    Y = np.full((nPed,end_frame),np.nan)  # (pid,frame)
    for pid,tid,st,et,sx,ex,sy,ey in tqdm(zip(df['pedestrianId'],df['targetId-PID2'],df['simTime'],df['endTime-PID1'],df['startX-PID1'],df['endX-PID1'],df['startY-PID1'],df['endY-PID1']),total=df.shape[0]):
        pid = int(pid)
        start_fr = max(int(round(st/frame_skip)),start_frame)
        end_fr = min(int(round(et/frame_skip))+1,end_frame)
        if end_fr-start_fr>0:
            X[pid,start_fr:end_fr] = np.linspace(sx,ex, end_fr-start_fr)
            Y[pid,start_fr:end_fr] = np.linspace(sy,ey, end_fr-start_fr)
    return(scale*X,scale*Y)


# Adding Gaussian noise
def add_noise_to_coords(X,Y,std):
    return(X+np.random.normal(0,std,size=(X.shape)),
           Y+np.random.normal(0,std,size=(Y.shape)))

def compute_velocity(X,Y, epsilon):
    # Delete singular points on simulation data(entrance, exit)
    del_idx = np.where( np.logical_or(Y > 45, Y < 5) )
    X[del_idx] = np.nan
    Y[del_idx] = np.nan
        
    # velocity data
    df_u = np.roll(X,1)-X
    df_v = np.roll(Y,1)-Y
    df_norm = np.sqrt(df_u**2 + df_v**2)
            
    # If velocity is equal to 0(or less than std of noise), it takes former velocity
    for j in range(len(df_u[0])):
        small_v_idx = np.where(df_norm[:,j] <= epsilon)
        if len(small_v_idx[0])>0:
            df_u[small_v_idx,j] = df_u[small_v_idx,j-1]
            df_v[small_v_idx,j] = df_v[small_v_idx,j-1]
    return(df_u, df_v)

## u[ped,frame]
def compute_moment(u,v,p):
    if u.shape[0]==0:
        return(np.zeros(1))
    df_norm = np.sqrt(u**2 + v**2)
    T = ((u/df_norm + v/df_norm * 1j)**p).mean(axis=1)
    return(np.absolute(T))

def entropy_base(u,v):
    if u.shape[0]==0:
        return(np.zeros(1))
    df_norm = np.sqrt(u**2 + v**2)
    ent = scipy.stats.entropy(df_norm,axis=1)
    return(np.maximum(0,1-ent/np.log(u.shape[1])))
    #return(np.where(np.isnan(ent), 0, 1-ent/np.log(u.shape[1])))
    
def moving_average(values, alpha=0.8):
    indicator_list_ma = []
    indicator_list_ma.append(values[0])
    for i in range(len(values)-1):
        indicator_list_ma.append(alpha*values[i+1] + (1-alpha)*indicator_list_ma[i])
    return(indicator_list_ma)
# not used
def compute_R1(u,v):
    if u.shape[0]==0:
        return(np.zeros(1))
    df_norm = np.sqrt(u**2 + v**2)
    Tu = u/df_norm
    Tv = v/df_norm
    return(np.sqrt(np.mean(Tu,axis=1)**2 + np.mean(Tv,axis=1)**2))

# not used
def compute_R2(u,v):
    if u.shape[0]==0:
        return(np.zeros(1))
    df_norm = np.sqrt(u**2 + v**2)
    Tu = u/df_norm
    Tv = v/df_norm
    Tu2 = Tu**2 - Tv**2
    Tv2 = 2*Tu*Tv
    return(np.sqrt(np.mean(Tu2,axis=1)**2 + np.mean(Tv2,axis=1)**2))



    