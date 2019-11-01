import torch
import torch.nn as nn

#NO NUMPY

def cov(t):
  t_exp=torch.mean(t, dim=1).double()
  print(t_exp)
  t_norm=(t-t_exp[:, None]).double()
  
  one=1.
  covariance=one/(t.size(1)-one)*t_norm.mm(t_norm.t())
  return covariance.double()

class Gaussian_Fit(nn.Module):
   def __init__(self, activation_maps):
    super(Gaussian_Fit, self).__init__()
 
   def forward(self, x):
    #x would be the activation maps generated by pose encoder
    x_shape=list(x.size())
    assert len(x_shape)==4, "input is not a 4-tensor"
    print("x")
    print(x)
    mu=x.mean(3)
    mu=mu.mean(2)
    print("mu")
    print(mu)
  
    ones=torch.empty(x_shape[0], x_shape[1],x_shape[2], x_shape[3], dtype=torch.float)
    covs=torch.empty(x_shape[0], x_shape[1],x_shape[2], x_shape[3], dtype=torch.float)
    inv_covs=torch.empty(x_shape[0], x_shape[1],x_shape[2], x_shape[3], dtype=torch.float)
    approx=torch.empty(x_shape[0], x_shape[1],x_shape[2], x_shape[3], dtype=torch.float)

    ones.fill_(1.)
    mu_tensor=ones

    for i in range(x_shape[0]):
      for j in range(x_shape[1]):
        x_sum=torch.sum(x[i][j])
        x[i][j]/=x_sum
        mu[i][j]/=x_sum
        mu_tensor[i][j]*=mu[i][j]
        x[i][j]-=mu[i][j]
        #print("\n"+"x[{}][{}]".format(i, j))
        #print(x[i][j])
        covs[i][j]=cov(x[i][j])
    
        epsilon=torch.randn(x_shape[2], x_shape[3])/1e7
        #print("epsilon")
        #print(epsilon)
        covs[i][j]+=epsilon
        #print("\n"+"covs[{}][{}]".format(i, j))
        #print(covs[i][j])
        inv_covs[i][j]=covs[i][j].inverse()
        #print("\n"+"inv_covs[{}][{}]".format(i, j))
        #print(inv_covs[i][j])
    
        approx[i][j]=torch.mm(torch.mm(x[i][j], inv_covs[i][j]), x[i][j])
        #print("\n"+"approx[{}][{}]".format(i, j))
        #print(approx[i][j])
        approx[i][j]+=1
        #print(approx[i][j])
        approx[i][j]=approx[i][j].pow(-1)
        #print("approx")
        #print(approx[i][j])

    return approx