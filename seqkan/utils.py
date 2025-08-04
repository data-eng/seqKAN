import numpy
import torch
import matplotlib.pyplot as plt

import seqkan.kan.spline

import code


global counter
counter = 0


'''
Plots specific datapoints on a spline
'''

def plot_batch( x, grid, coef, k, outdir ):
    global counter
    #code.interact( local=locals() )
    left = grid[0,0]
    right = grid[0,-1]
    # Reminder: k=1 for the inactivated hidden layer
    if k == 3:
        xx = numpy.linspace( left, right, 128 )
        yy = seqkan.kan.spline.coef2curve( torch.Tensor(xx.reshape(128,1)),
                                           grid, coef, k )
        plt.plot( xx, yy[:,0].detach().numpy() )

        y = seqkan.kan.spline.coef2curve( x, grid, coef, k )
        #code.interact( local=locals() )
        plt.scatter( x[:,0].detach().numpy(), y[:,0,0].detach().numpy(), marker='x' )
        plt.savefig( f"out/q_{counter}.png" )
        plt.close()
        counter += 1
