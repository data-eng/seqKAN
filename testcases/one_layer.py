import os
import time
import code

import numpy
import matplotlib.pyplot as plt

import torch.nn
import torch.optim
import torch.utils.data

import seqkan
from testcases import testcase_data


outdir = "out/"
os.makedirs(outdir, exist_ok=True)


#dd = testcase_data.Periodic( -10, 10, 512, seq=False )
dd = testcase_data.Sequence( 512, seq=False )

trn_dataset = torch.utils.data.Subset( dd, range(0, dd.dsize*7//8) )
val_dataset = torch.utils.data.Subset( dd, range(dd.dsize*7//8, len(dd)) )


kan_params = {
    "output": { "k": 3, "grid": 31, "grid_range": dd.range }
}

prc = seqkan.kan.KANLayer( in_dim=1, out_dim=1,
                           grid_range=kan_params["output"]["grid_range"],
                           num=kan_params["output"]["grid"],
                           k=kan_params["output"]["k"],
                           noise_scale=dd.output_scale,
                           scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                           grid_eps=0.02, sp_trainable=True )

# Not having this gives a RuntimeError from deep inside torch.
# TODO: Try if turning coef, grid into float32 also does the trick.
prc.double()

batch_size = 16
epochs = 20
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam( prc.parameters(), lr=0.05 )
optimizer = torch.optim.SGD( prc.parameters(), lr=0.2 )

trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

## Plotting

all_n = 128
all_t, all_x, tru_y = dd.plotting_helper( all_n )


if 1==0:
    onego_coef = seqkan.kan.spline.curve2coef(
        torch.Tensor(all_x).unsqueeze(dim=1),
        torch.Tensor( tru_y.reshape(128,1,1) ),
        prc.grid, kan_params["output"]["k"] )
    onego_y = seqkan.kan.spline.coef2curve(
        torch.Tensor(all_x).unsqueeze(dim=1),
        prc.grid, onego_coef, kan_params["output"]["k"] )
    onego_y = onego_y.squeeze().numpy()
    plt.plot( all_x, tru_y )
    plt.plot( all_x, onego_y )
    plt.savefig( f"{outdir}/q_one_go" )
    plt.close()

old_coef = prc.coef.clone()
training_time = 0
validation_time = 0

for epoch in range(epochs):

    prc.train()
    batch = 0
    for x, y in trn_loader:
        start_time = time.time()
        #code.interact( local=locals() )
        y_pred = prc( x.unsqueeze(dim=1) )
        loss = criterion(y_pred[0].squeeze(), y.squeeze())
        #code.interact( local=locals() )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += (time.time() - start_time)

        all_xx = torch.Tensor( all_x.reshape(128,1) )
        all_yy = seqkan.kan.spline.coef2curve( all_xx,
                                               prc.grid,
                                               prc.coef, kan_params["output"]["k"] )
        #code.interact( local=locals() )
        #plt.ylim( (-1.5,1.5) )
        plt.plot( all_x, all_yy.squeeze().detach().numpy() )

        plt.scatter( x.detach().numpy(), y.detach().numpy() )
        plt.plot( all_x, tru_y )

        #for i in range(len(x)):
        #    plt.annotate( label, (x[i], y[i]), textcoords="offset points", xytext=(5,5), ha='left')

        l = prc.grid.shape[1]
        # for k == 3
        for i in range(3-1,l-3+1):
            x = prc.grid.detach().numpy()[0,i]
            y = 10*(prc.coef.detach().numpy()[0,0,i-2] - old_coef.detach().numpy()[0,0,i-2])
            plt.plot( [x,x], [0,y] )

        plt.savefig( f"{outdir}/q_{epoch}_{batch}" )
        plt.close()
        old_coef = prc.coef.clone()
        batch += 1
        # for batch

    prc.eval()
    trn_loss = 0.0
    with torch.no_grad():
        for x,y in trn_loader:
            yy = prc( x.unsqueeze(dim=1) )
            trn_loss += criterion(yy[0].squeeze(), y)
    print( f"Epoch {epoch}: loss {trn_loss}, training time: {training_time}" )
    # for epoch

prc.eval()

val_loss = 0.0
with torch.no_grad():
    for x,y in trn_loader:
        start_time = time.time()
        yy = prc( x.unsqueeze(dim=1) )
        validation_time += (time.time() - start_time)
        val_loss += criterion(yy[0].squeeze(), y)
print( f"Training time: {training_time}" )
print( f"Inf time/loss on training data: {val_loss}" )

val_loss = 0.0
with torch.no_grad():
    for x,y in val_loader:
        start_time = time.time()
        yy = prc( x.unsqueeze(dim=1) )
        validation_time += (time.time() - start_time)
        val_loss += criterion(yy[0].squeeze(), y)

print( f"Validation time: {validation_time}" )
print( f"Validation loss: {val_loss}" )


    
'''
prc.eval()

xx = torch.Tensor( x.reshape(128,1) )

grid1 = prc.KANlayer.act_fun[0].grid
coef1 = prc.KANlayer.act_fun[0].coef
plt.plot( x, yy[:,0].detach().numpy() )
plt.plot( dd.x, dd.y )
plt.savefig( outdir + "plot1.png" )
plt.close()

grid2 = prc.KANoutput.act_fun[0].grid
coef2 = prc.KANoutput.act_fun[0].coef
yy = seqkan.kan.spline.coef2curve( xx, grid2, coef2, kan_params["output"]["k"] )
plt.plot( x, yy[:,0].detach().numpy() )
plt.plot( dd.x, dd.y )
plt.savefig( outdir + "plot2.png" )
plt.close()
'''
