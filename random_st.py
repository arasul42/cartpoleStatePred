import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
 

# process disturbance
# mean
mP=np.array([0,0])
# covariance
cP=np.array([[0.0001, 0],[0, 0.0001]])
 
# measurement noise
# mean
mN=np.array([0])
# covariance
cN=np.array([[0.001]])
 
 
# create distributions 
# process disturbance distribution
pDistribution=multivariate_normal(mean=mP,cov=cP)
# noise distribution
nDistribution=multivariate_normal(mean=mN,cov=cN)

pDistribution.rvs(size=1).reshape(2,1)

# measurement noise
# mean
mN=np.array([0])
# covariance
cN=np.array([[0.001]])
 

# create distributions 
# process disturbance distribution
pDistribution=multivariate_normal(mean=mP,cov=cP)
# noise distribution
nDistribution=multivariate_normal(mean=mN,cov=cN)
 
 
# Construct a continuous-time system
m=5
ks=200
kd=30
 
Ac=np.array([[0,1],[-ks/m, -kd/m]])
Cc=np.array([[1,0]])
Bc=np.array([[0],[1/m]])
 
# discretize the system
# discretization constant
h=0.005


A=np.linalg.inv(np.eye(2)-h*Ac)
B=h*np.matmul(A,Bc)
C=Cc


# select the initial state
x0=np.array([[0.1],[0.01]])


simSteps = 1500  # or any desired number of simulation steps

 
# control input
#controlInput=10*np.random.rand(1,simTime)
cI=100*np.ones((1,simSteps))
 
# zero-state trajectory
sT=np.zeros(shape=(2,simSteps+1))
 
# output 
output=np.zeros(shape=(1,simSteps))
 
# set the initial state
sT[:,[0]]=x0
 
# simulate the state-space model
for i in range(simSteps):
    sT[:,[i+1]]=np.matmul(A,sT[:,[i]])+np.matmul(B,cI[:,[i]])+pDistribution.rvs(size=1).reshape(2,1)
    output[:,[i]]=np.matmul(C,sT[:,[i]])+nDistribution.rvs(size=1).reshape(1,1)

# create a time vector
timeVector=np.linspace(0,(simSteps-1)*h,simSteps)
 
# plot the time response
plt.figure(figsize=(10,8))
plt.plot(timeVector,sT[0,0:simSteps], color='blue',linewidth=4)
plt.title("State", fontsize=14)
plt.xlabel("time", fontsize=14)
plt.ylabel("State",fontsize=14)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.grid(visible=True)
plt.savefig("stateTrajectoryTime.png",dpi=600)
plt.show()
 
# plot the state-space trajectory
plt.figure(figsize=(10,8))
plt.plot(sT[0,0:simSteps],sT[1,0:simSteps], color='blue',linewidth=4,alpha=0.5)
plt.title("State", fontsize=16)
plt.xlabel("x1", fontsize=16)
plt.ylabel("x2",fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.grid(visible=True)
plt.scatter(sT[0,0], sT[1,0], s=500,c='r', marker='o',label='Start' )
plt.scatter(sT[0,-1], sT[1,-1], s=500,c='k', marker='o',linewidth=6,label='End' )
plt.legend(fontsize=14)
plt.savefig("stateTrajectory.png",dpi=600)
plt.show()

