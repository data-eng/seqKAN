import os

import numpy
import torch
import matplotlib.pyplot as plt

import seqkan
from testcases import testcase_data

outdir = "out/"
os.makedirs(outdir, exist_ok=True)

#dd = testcase_data.Periodic( -10, 10, 512, seq=False )
dd = testcase_data.Sequence( 512, seq=False )

all_instances = torch.utils.data.DataLoader( dd, batch_size=dd.dsize, shuffle=False )

kan_params = {
    "output": { "k": 3, "grid": 5, "grid_range": dd.range }
}

prc = seqkan.kan.KANLayer( in_dim=1, out_dim=1,
                           grid_range=kan_params["output"]["grid_range"],
                           num=kan_params["output"]["grid"],
                           k=kan_params["output"]["k"],
                           noise_scale=dd.output_scale,
                           scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                           grid_eps=0.02, sp_trainable=True )

grid = prc.grid
x, y = next(iter(all_instances))
xx = x.detach().unsqueeze( dim=1 )
yy = y.detach().reshape(dd.dsize,1,1)
coef = seqkan.kan.spline.curve2coef( xx, yy,
                                     grid, kan_params["output"]["k"] )
yy1 = seqkan.kan.spline.coef2curve( xx, grid,
                                    coef, kan_params["output"]["k"] )
plt.plot( x.reshape(dd.dsize), yy1[:,0].squeeze().detach().numpy() )
plt.savefig("aa1.png")
plt.close()

plt.plot( x.reshape(dd.dsize), y.reshape(dd.dsize) )
plt.savefig("aa2.png")
plt.close()

