import os
import code

import numpy
import torch
import matplotlib.pyplot as plt

import seqkan
from testcases import testcase_data

import scipy.optimize


outdir = "out/"
os.makedirs(outdir, exist_ok=True)


##
## Setup the data
##


if 1==1:
    dd = testcase_data.Periodic( -10, 10, 1024, seq=True )
    params = {
        "0": { "k": 3, "grid":  21, "grid_range": dd.range },
        "1a": { "k": 3, "grid": 31, "grid_range": dd.range },
        "1b": { "k": 3, "grid": 11, "grid_range": dd.range }
    }

if 1==0:
    dd = testcase_data.Sequence( 1024, seq=True )
    params = {
        "0": { "k": 3, "grid":  21, "grid_range": dd.range },
        "1a": { "k": 3, "grid": 31, "grid_range": dd.range },
        "1b": { "k": 3, "grid": 11, "grid_range": dd.range }
    }

if 1==0:
    dd = testcase_data.Detrending( 2048, seq=True )
    params = {
        "0": { "k": 3, "grid":  21, "grid_range": dd.range },
        "1a": { "k": 3, "grid": 31, "grid_range": dd.range },
        "1b": { "k": 3, "grid": 31, "grid_range": dd.range }
    }


all_instances = torch.utils.data.DataLoader( dd, batch_size=dd.dsize, shuffle=False )

global x, y
x, y = next(iter(all_instances))
# Keep only the last element of each sequence,
# bit do not squeeze the third dimension
x = x[:,-1,:]


##
## Setup the model
##


prc0 = seqkan.kan.KANLayer( in_dim=1, out_dim=1,
                            grid_range=params["0"]["grid_range"],
                            num=params["0"]["grid"],
                            k=params["0"]["k"],
                            noise_scale=dd.output_scale,
                            scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                            grid_eps=0.02, sp_trainable=True )
prc0.double()

prc1a = seqkan.kan.KANLayer( in_dim=1, out_dim=1,
                             grid_range=params["1a"]["grid_range"],
                             num=params["1a"]["grid"],
                             k=params["1a"]["k"],
                             noise_scale=dd.output_scale,
                             scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                             grid_eps=0.02, sp_trainable=True )
prc1a.double()

prc1b = seqkan.kan.KANLayer( in_dim=1, out_dim=1,
                             grid_range=params["1b"]["grid_range"],
                             num=params["1b"]["grid"],
                             k=params["1b"]["k"],
                             noise_scale=dd.output_scale,
                             scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                             grid_eps=0.02, sp_trainable=True )
prc1b.double()


coefs = torch.cat(
    ( prc0.coef.clone().detach().squeeze(),
      prc1a.coef.clone().detach().squeeze(),
      prc1b.coef.clone().detach().squeeze() ) )


def fun( x, coef ):
    p0 = params["0"]["grid"] + 3
    c0 = torch.tensor( coef[0:p0].copy().reshape(1,1,p0) )
    x1 = seqkan.kan.spline.coef2curve( x, prc0.grid,
                                       c0,
                                       params["0"]["k"] )
    p1 = p0 + params["1a"]["grid"] + 3
    c1a = torch.tensor( coef[p0:p1].copy().reshape(1,1,p1-p0) )
    #code.interact( local=locals() )
    x2a = seqkan.kan.spline.coef2curve( x1.reshape(dd.dsize,1),
                                        prc1a.grid,
                                        c1a,
                                        params["1a"]["k"] )
    p2 = p1 + params["1b"]["grid"] + 3
    c1b = torch.tensor( coef[p1:p2].copy().reshape(1,1,p2-p1) )
    assert p2 == len(coef)
    x2b = seqkan.kan.spline.coef2curve( x1.reshape(dd.dsize,1),
                                        prc1b.grid, c1b,
                                        params["1b"]["k"] )

    return x1, x2a, x2b


def plot_fun( coef, x1, x2a, x2b ):
    global x,y 
    fig, ax = plt.subplots( 2, 2 )

    ax[0,0].plot( x.reshape(dd.dsize), y[:,0].squeeze().detach().numpy(),
                  color="green" )
    ax[0,0].plot( x.reshape(dd.dsize), x2a.squeeze().detach().numpy(),
                  color="red" )
    ax[0,1].plot( x.reshape(dd.dsize), y[:,1].squeeze().detach().numpy(),
                  color="green" )
    ax[0,1].plot( x.reshape(dd.dsize), x2b.squeeze().detach().numpy(),
                  color="red" )
    ax[1,0].plot( x.reshape(dd.dsize), x1.squeeze().detach().numpy(),
                  color="red" )

    q = torch.cat( (x1[:,:,0],x2a[:,:,0],x2b[:,:,0]), dim=1 )
    x1x2 = torch.sort( q, dim=0 ).values
    ax[1,1].plot(x1x2[:,0],x1x2[:,1:3])
    
    #plt.plot( x.reshape(dd.dsize), x2a.squeeze().detach().numpy() )

    #plt.plot( x.reshape(dd.dsize), y[:,1].squeeze().detach().numpy() )
    #plt.plot( x.reshape(dd.dsize), x2b.squeeze().detach().numpy() )
    plt.savefig("bare_grid.png")
    plt.close()

    #plt.plot( x.reshape(dd.dsize), x2a.squeeze().detach().numpy() )
    #plt.plot( x.reshape(dd.dsize), x2b.squeeze().detach().numpy() )
    

def evaluate( coef ):
    global x,y 
    _, x2a, x2b = fun( x, coef )
    y_pred = torch.Tensor( numpy.array([x2a[:,0,0],x2b[:,0,0]]).T )
    l = 0;
    # If there is only one y, duplicate it
    if y.shape[1] == 1:
        y = y.repeat(1,2)
    for i in range(y.shape[1]):
        l += torch.mean( (y_pred[:,i]-y[:,i])**2 )
    return l.item()


def learn():
    #l = evaluate( coefs.numpy() )
    res = scipy.optimize.minimize( evaluate, coefs,
                                   method="BFGS", options={"disp": True} )
    #l2 = evaluate( res.x )
    x1, x2a, x2b = fun( x, res.x )
    return res.x, x1, x2a, x2b

new_coefs, x1, x2a, x2b = learn()
plot_fun( new_coefs, x1, x2a, x2b )
