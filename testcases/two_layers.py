import os
import code
import time

import numpy
import matplotlib.pyplot as plt

import torch.nn
import torch.optim
import torch.utils.data

import seqkan
from testcases import testcase_data


outdir = "out/"
os.makedirs(outdir, exist_ok=True)


#dd = testcase_data.Periodic( -10, 10, 512, seq=True )
dd = testcase_data.Sequence( 512, seq=True )

trn_dataset = torch.utils.data.Subset( dd, range(dd.dsize*1//8, dd.dsize) )
val_dataset = torch.utils.data.Subset( dd, range(0, dd.dsize*1//8) )

print( dd.output_scale )

## Construct the seqKAN object

kan_params = {
    "hidden": { "k": 3, "grid": 11, "grid_range": [dd.left,dd.right] },
    "output": { "k": 3, "grid": 11, "grid_range": [dd.left,dd.right],
                "noise_scale": dd.output_scale }
}

prc = seqkan.seqKAN( 1, 1, 1, kan_params=kan_params, device="cpu" )

# Not having this gives a RuntimeError from deep inside torch.
# TODO: Try if turning coef, grid into float32 also does the trick.
prc.double()


## Plotting

all_n = 128
all_t, all_xin, all_xout = dd.plotting_helper( all_n )


## Training Loop

batch_size = 16
epochs = 5
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam( prc.parameters(), lr=0.05 )
optimizer = torch.optim.SGD( prc.parameters(), lr=0.2 )

trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=dd.dsize, shuffle=False)

old_coef = prc.KANlayer.act_fun[0].coef.clone()
training_time = 0
time_fwd = 0
time_loss = 0
time_bp = 0
time_opt = 0

old_coef0 = prc.KANlayer.act_fun[0].coef.clone()
old_coef1 = prc.KANoutput.act_fun[0].coef.clone()

for epoch in range(epochs):

    prc.train()
    batch = 0
    for x, y in trn_loader:

        xx = x.detach().numpy()
        yy = y.detach().squeeze().numpy()

        start_time = time.time()
        y_pred = prc( x.unsqueeze(dim=1) )
        time_fwd1 = time.time()
        loss = criterion(y_pred.squeeze(), y.squeeze())
        time_loss1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        time_bp1 = time.time()
        optimizer.step()
        time_opt1 = time.time()

        training_time += (time.time() - start_time)
        time_fwd += (time_fwd1 - start_time)
        time_loss += (time_loss1 - time_fwd1)
        time_bp += (time_bp1 - time_loss1)
        time_opt += (time_opt1 - time_bp1)

        fig, ax = plt.subplots( 2, 3 )

        all_xin = torch.Tensor( all_xin.reshape(all_n,1) )
        all_prd = seqkan.kan.spline.coef2curve( all_xin,
                                               prc.KANlayer.act_fun[0].grid,
                                               prc.KANlayer.act_fun[0].coef,
                                               kan_params["output"]["k"] )
        np_prd = all_prd.squeeze().detach().numpy()
        ax[0,0].plot( all_xin, np_prd, color="red" )
        ax[0,0].plot( all_xin, np_prd.sum(axis=1), color="blue" )
        ax[0,0].plot( all_xin, all_xout, color="green" )
        ax[0,0].scatter( xx, yy )

        l = prc.KANlayer.act_fun[0].grid.shape[1]
        # for k == 3
        for i in range(3-1,l-3+1):
            x = prc.KANlayer.act_fun[0].grid.detach().numpy()[0,i]
            y = dd.output_scale*(prc.KANlayer.act_fun[0].coef.detach().numpy()[0,0,i-2] - old_coef0.detach().numpy()[0,0,i-2])
            ax[1,0].plot( [x,x], [0,y] )
        old_coef0 = prc.KANlayer.act_fun[0].coef.clone()

        # Feed the previous output into the next layer,
        #code.interact( local=locals() )
        all_prd = all_prd.squeeze()
        all_prd2 = seqkan.kan.spline.coef2curve( all_prd,
                                                 prc.KANoutput.act_fun[0].grid,
                                                 prc.KANoutput.act_fun[0].coef,
                                                 kan_params["output"]["k"] )
        all_prd = all_prd.sum(axis=1).detach().numpy()
        all_prd2 = all_prd2.sum(axis=1).detach().numpy()
        # scatter, not plot, because all_prd is not sorted
        ax[0,1].scatter( all_prd, all_prd2, color="red" )
        ax[0,1].scatter( all_prd, all_prd2.sum(axis=1), color="blue" )
        ax[0,1].scatter( all_prd, all_xout, color="green" )


        # Feed the previous output into the next layer,
        # but plot against inputs
        #code.interact( local=locals() )
        ax[0,2].plot( all_xin, all_prd2, color="red" )
        ax[0,2].plot( all_xin, all_prd2.sum(axis=1), color="blue" )
        ax[0,2].plot( all_xin, all_xout, color="green" )
        ax[0,2].scatter( xx, yy )

        # Plot the coef updates
        
        l = prc.KANoutput.act_fun[0].grid.shape[1]
        # for k == 3
        for i in range(3-1,l-3+1):
            x = prc.KANoutput.act_fun[0].grid.detach().numpy()[0,i]
            y = dd.output_scale*(prc.KANoutput.act_fun[0].coef.detach().numpy()[0,0,i-2] - old_coef1.detach().numpy()[0,0,i-2])
            ax[1,1].plot( [x,x], [0,y] )
        old_coef1 = prc.KANoutput.act_fun[0].coef.clone()

        plt.savefig( f"{outdir}/q_{epoch}_{batch}" )
        plt.close()
        batch += 1
        # for batch

    prc.eval()

    trn_loss = 0.0
    mse = 0.0
    with torch.no_grad():
        for xx, yy in val_loader:
            xout = prc( xx.unsqueeze(dim=1) )
            trn_loss += criterion( xout.squeeze(), yy.squeeze() )
            mse += torch.mean( (xout.squeeze()-yy)**2 )
    print( f"Epoch {epoch}: loss {trn_loss} {mse}" )
    print( f"Training time, fwd pass {time_fwd:0.3f},  calc loss {time_loss:0.3f}, bp {time_bp:0.3f}, opt {time_opt:0.3f}, total: {training_time:0.3f}" )
    # for epoch
