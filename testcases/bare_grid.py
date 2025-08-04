import numpy
import torch
import matplotlib.pyplot as plt

import seqkan

dim = 1
num = 31
k = 3

left = 10
right = 40

model = seqkan.kan.KANLayer( in_dim=dim, out_dim=dim, num=num, grid_range=[left,right] )
grid = model.grid
coef = model.coef

dsize = 2048
x = numpy.linspace( left, right, dsize )
q = numpy.floor( x/10 )
y = numpy.sin( q*x )

x = numpy.array(x).reshape(dsize,1)
xx = torch.Tensor( x )
y = numpy.array(y).reshape(dsize,1,1)
yy = torch.Tensor( y )
coef = seqkan.kan.spline.curve2coef( xx, yy, grid, k )

yy1 = seqkan.kan.spline.coef2curve( xx, grid, coef, k )
plt.plot( x.reshape(dsize), yy1[:,0].squeeze().detach().numpy() )
plt.savefig("aa1.png")
plt.close()

plt.plot( x.reshape(dsize), y.reshape(dsize) )
plt.savefig("aa2.png")
plt.close()

