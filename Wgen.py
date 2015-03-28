### Generate weights
### Paramters:   
###				N = size of network
###				Ng = number of neurons in each group.  Length(Ng) = number of groups
###				mu_g = means for the groups: a Length(Ng)x1 array, with means > 0 first in the array
###				var_g = variances for the group/group interactions: a Length(Ng)xLength(Ng) matrix
###                             fixed = true if the sum of each row should equal 0, false otherwise

import numpy as np

def get_weights(N, Ng, mu_g, var_g, fixed):
    mu_g = np.array(mu_g)
    var_g = np.array(var_g)
	
    Ng.insert(0, 0)
    ngroups = np.size(Ng)
    weights = np.zeros((N, N))
    inds = np.cumsum(Ng)
    for group in range(ngroups-1):
        for group2 in range(ngroups-1):
            submat = np.sqrt(var_g[group, group2])*np.random.randn(Ng[group+1], Ng[group2+1]) + mu_g[group2]
            weights[inds[group]:inds[group+1], inds[group2]:inds[group2+1]] = submat
    
    if fixed:
        for row in range(np.size(weights, axis=1)):
            weights[row,:] = weights[row,:] - (np.sum(weights[row,:])/N)
    weights[weights[:,0:np.sum(Ng[0:np.size(mu_g[mu_g>0])+1])] < 0] = 0
    weights[weights[:,np.sum(Ng[0:np.size(mu_g[mu_g>0])+1]):np.sum(Ng[np.size(mu_g[mu_g>0])+1:np.size(Ng)])] > 0] = 0

    return weights