# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import pylab as plt
from Wgen import get_weights
import scipy.fftpack as spfft 
from numpy import linalg as LA

def get_learn_index(t, z, freq):
    zf = np.abs(spfft.fft(z))
    freqs = spfft.fftfreq(np.size(zf), (t[1]-t[0])/1000)

    zf_plot = zf[0:np.size(t)/2] 
    freqs_plot = freqs[0:np.size(t)/2]
    
    num_f = (zf_plot[max([i for i,j in enumerate(freqs_plot) if j<=freq])])**2

    sum_f = np.sum(zf_plot)**2

    return num_f/sum_f
    
targ_freq = 10 #Hz
def f(t):
    return 1.5*np.sin(2*np.pi*targ_freq*t/1000)
    
Ng = 500.0 # Network Size
tau = 10  # time constant
alpha = 1.0 # "Learning Rate"

ggg = 1.5
pgg = 0.1
ggz = 1
alpha = 1
pgg = 0.1

sigma_w = np.sqrt(1/(Ng))
Jgz = np.random.uniform(-1, 1, Ng)
w = np.random.randn(Ng)*sigma_w

x = np.random.randn(Ng)*sigma_w
P = np.identity(Ng) / alpha
P = np.reshape(P, Ng*Ng)
print "Creating statevector"
svinit =  np.concatenate((x, w, P))
print "Done"

def dNeurons(statevec, t, param):
    print t
    print gamma
    print trial
    dt = param[0]
    training = param[1]
    f = param[2]
    
    x_i = statevec[0:Ng]
    w_i = statevec[Ng:2*Ng]
    P_i = statevec[2*Ng:]
    P_i = np.reshape(P_i, (Ng, Ng))
    
    r_i = np.tanh(x_i)
    
    z_i = np.dot(w_i, r_i)
    Iinj = 0.0
        
    if training > 0 and t > training and t < training+train_dur:
        
        dxidt = (Iinj -x_i + ggg*np.dot(Jgg, r_i) + ggz*np.dot(Jgz, z_i))/tau
        x_new = x_i + dxidt*dt
        r_new = np.tanh(x_new)
        
        eminus = np.dot(w_i, r_new) - f(t) 
        denom = (1 + np.dot(np.transpose(r_new), np.dot(P_i, r_new)))
        dPdt = -1*np.outer(np.dot(P_i, r_new), np.dot(np.transpose(r_new), P_i)) / denom
        P_new = P_i + dPdt*dt
        dwdt = -1*eminus*np.dot(P_new, r_new)
        w_new = w_i + dwdt*dt
        eplus = np.dot(w_new, r_new) - f(t)
        
    else:
        dxidt = (Iinj -x_i + ggg*np.dot(Jgg, r_i) + ggz*np.dot(Jgz, z_i))/tau
        x_new = x_i + dxidt*dt
        r_new = np.tanh(x_new)
        dwdt = np.zeros(np.shape(w_i))
        w_new = w_i
        P_new = P_i
    
    P_new = np.reshape(P_new, Ng*Ng)
    return np.concatenate((x_new, w_new, P_new))
    
# Run the network
dt = 1
training = 2000
train_dur = 1000*10
dur = training+train_dur+3000
times = sp.arange(0.0, dur, dt)
samps = np.size(times)

params = (dt, training, f)
xsave = np.zeros((samps, Ng))
wsave = np.zeros((samps, Ng))

fis = [0.02]

epsilon = 0.2
var_oo = var_oi = var_io = var_ii = ((1.-epsilon)**2)/(Ng)

gammas = sp.arange(1.-epsilon + 0.1, 8.0, 0.5) 

num_trials = 3
learn_index_save = np.zeros((np.size(gammas), num_trials))

for fi in fis:
    for ind, gamma in enumerate(gammas):
        fi = 0.2
        fe = 1. - fi

        fe_old = fe*0.985
        fe_new = fe - fe_old
        
        Ne_old = np.floor(Ng*fe_old)
        Ne_new = np.floor(Ng*fe_new)
        Ni = Ng - Ne_old - Ne_new

        #ues = sp.arange(0.25, 0.25, 0.01)
        ue_old = 1.
        ue_new = 1.
        ui = -(fe_old*ue_old+fe_new*ue_new)/fi
    
        for trial in range(0, num_trials):
            Ngs = [Ne_old, Ne_new, Ni]
            var_nn = var_no = var_ni = var_on = var_in = (gamma**2)/(Ng)
            vars_all = [[var_oo, var_no, var_io], [var_on, var_nn, var_in], [var_oi, var_ni, var_ii]]
            means_all = [ue_old, ue_new, ui]
    
            Jgg = get_weights(Ng, Ngs, means_all, vars_all, True)
    
            evals, evec = LA.eig(Jgg);
            evalx = sp.real(evals);
            evaly = sp.imag(evals);
            
            plt.figure()
            plt.plot(evalx, evaly, 'b.')
            plt.axis('equal')
            plt.xlabel('Re(lambda)')
            plt.ylabel('Im(lambda)')
            title_string = 'Spectrum, gamma = %.2f, fi = %.2f,ue = %.2f, n = %d, sums balanced' %(gamma, fi,ue_old, trial)
            fig_string = title_string+'.png'
            plt.title(title_string)
            plt.plot((1, 1), (-2, 2), 'k-')
            #plt.show()
            plt.savefig(fig_string)
            
            for indx, t in enumerate(times):
                xsave[indx, :] = svinit[0:Ng]
                wsave[indx, :] = svinit[Ng:2*Ng]
                svinit = dNeurons(svinit, t, params)
    
            rates = np.tanh(xsave)
    
            z = np.sum(wsave*rates, axis=1)
            learn_index_save = get_learn_index(times[training+train_dur:samps], z[training+train_dur:samps], targ_freq)
            print(learn_index_save)
            str_to_write = '%.1f\t%d\t%.9f\n' %(gamma, trial, learn_index_save)
            save_file = open('Learning Indices, vary fi and gamma, ue = 5, sums balanced.txt', 'a')
            save_file.write(str_to_write)
            save_file.close()
            
            plt.figure()
            plt.plot(z, 'b-', label = 'z')
            plt.plot(f(times), 'r-', label = 'target')
            plt.legend(loc = 'lower left')
            title_string = 'Plot, gamma = %.2f, fi = %.2f,ue = %.2f, n = %d, sums balanced' %(gamma, fi,ue_old, trial)
            plt.savefig(title_string+'.png')
            
learn_index_ave = np.average(learn_index_save, axis = 1)
learn_index_sem = np.std(learn_index_save, axis = 1)/np.sqrt(num_trials)