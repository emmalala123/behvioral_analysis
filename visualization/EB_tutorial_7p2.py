#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:42:32 2022

@author: emmabarash
"""
import numpy as np
import matplotlib.pyplot as plt

t_final = 3
dt = 0.1e-3
t0 = 0
t_vec = np.arange(t0,t_final,dt)

tau = 10e-3
r_max = 100
i_app = np.zeros(np.size(t_vec))   # applied current  

def circuit(w, tau, theta, r, r_max, i_app, t_vec, dt):
    for i in np.arange(1,np.size(t_vec)):
            w = np.array(w)
            iapp_i = i_app[i-1] + (np.matmul(w.T,r[:,i-1]))
            
            dRidt = (-r[:,i-1] + iapp_i - theta)/tau
            r[:,i] = r[:,i-1] + dRidt*dt
            
            r[:,i] = np.minimum(r_max,r[:,i])
            r[:,i] = np.maximum(0,r[:,i])
            
    return r, iapp_i

# connection frequencies for part one
theta = [-5,-10]
#i_app[6250:12500] = 1
i_app[6250:12500] = -1
# connection strengths for part one
w11 = 0.6 
w12 = 1
w21 = -0.2
w22 = 0
w = [[w11,w21],[w12,w22]]
r = np.zeros([2,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, theta, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2'])

# connection frequencies for part two
theta = [10,5]
# connection strengths for part two
w11 = 1.2
w12 = -0.3
w21 = -0.2
w22 = 1.1

w = [[w11,w21],[w12,w22]]
i_app[6250:12500] = 50
#i_app[6250:12500] = -50
r = np.zeros([2,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, theta, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2'])

# connection frequencies for part three
theta = [-10,0]
i_app[6250:12500] = 50
#i_app[6250:12500] = -1
# connection strengths for part three
w11 = 2.5
w12 = 2
w21 = -3.0
w22 = -2

w = [[w11,w21],[w12,w22]]
r = np.zeros([2,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, theta, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2'])


# connection frequencies for part four
theta = [-10,-10]
i_app[6250:12500] = -1
#i_app[6250:12500] = 1
# connection strengths for part four
w11 = 0.8
w12 = -0.2
w21 = -0.4
w22 = 0.6

w = [[w11,w21],[w12,w22]]
r = np.zeros([2,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, theta, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2'])


# connection frequencies for part five
theta = [0,20]
i_app[6250:12500] = 50
#i_app[6250:12500] = 50
# connection strengths for part five
w11 = 2
w12 = 1
w21 = -1.5
w22 = 0

w = [[w11,w21],[w12,w22]]
r = np.zeros([2,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, i, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2'])


# connection frequencies for part six
theta = [-10,-5,5]
i_app[6250:12500] = 50
#i_app[6250:12500] = -50
# connection strengths for part six
w11 = 1.5
w12 = 0
w13 = 1
w21 = 0
w22 = 2
w23 = 1
w31 = -2.5
w32 = -3
w33 = -1

w = [[w11,w21, w31],[w12,w22, w32],[w13,w23,w33]]
r = np.zeros([3,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, i, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2', 'unit 3'])


# connection frequencies for part seven
theta = [-18,-15,0]
i_app[6250:12500] = 50
#i_app[6250:12500] = -50
# connection strengths for part seven
w11 = 2.2
w12 = -0.5
w13 = 0.9
w21 = -0.7
w22 = 2
w23 = 1.2
w31 = -1.6
w32 = -1.2
w33 = 0

w = [[w11,w21, w31],[w12,w22, w32],[w13,w23,w33]]
r = np.zeros([3,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, i, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2', 'unit 3'])
    
# connection frequencies for part eight
theta = [-18,-15,0]
i_app[6250:12500] = -30
#i_app[6250:12500] = 30
# connection strengths for part eight
w11 = 2.05
w12 = -0.2
w13 = 1.2
w21 = -0.05
w22 = 2.1
w23 = 0.5
w31 = -1.6
w32 = -4
w33 = 0

w = [[w11,w21, w31],[w12,w22, w32],[w13,w23,w33]]
r = np.zeros([3,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, i, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2', 'unit 3'])


# connection frequencies for part nine
theta = [-10,-20,10]
i_app[6250:12500] = -50
#i_app[6250:12500] = 50
# connection strengths for part nine
w11 = 0.98
w12 = -0.015
w13 = -0.01
w21 = 0
w22 = 0.99
w23 = -0.02
w31 = -0.02
w32 = 0.005
w33 = 1.01

w = [[w11,w21, w31],[w12,w22, w32],[w13,w23,w33]]

r = np.zeros([3,len(t_vec)])
for idx,i in enumerate(theta):
    r, iapp_i = circuit(w, tau, i, r, r_max, i_app, t_vec, dt)
    
    plt.plot(r[idx,:])
    plt.title("firing rate vs time")
    plt.xlabel("Time (ms)")
    plt.ylabel("firing rate")
    plt.legend(['unit 1', 'unit 2', 'unit 3'])


