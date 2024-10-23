import numpy as np

def process_Synthetic(type,rho):
    n = 5000
    v = np.random.uniform(low=0.0, high=1, size=(n,))
    X = np.random.uniform(low=0.0, high=1, size=(n,6))
    z = np.random.uniform(low=0.0, high=1, size=(n,))
    if type == 'A':
        e = v*rho
        # Initialize treatment variable
        t = (np.sqrt(e * z) + e + z*z)/3
        y = t + e + (X[:,0]**2+X[:,1]+np.sqrt(X[:,2])+X[:,3]**2/2+X[:,4]/2+np.sqrt(X[:,5])/2)/6
    else:
        e = np.exp(rho*v-1)/rho
        # Initialize treatment variable
        t = (np.sqrt(e * z) + e + z*z)/3
        y = (np.exp(X[:,0])+X[:,1]+np.sqrt(X[:,2])+np.exp(X[:,3])/2+X[:,4]/2+np.sqrt(X[:,5])/2)/6*t + e
    return X,t,y,z

x,t,y,z = process_Synthetic('A',0.4)
print(t[:100])