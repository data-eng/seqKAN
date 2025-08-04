import collections.abc
import numpy
import torch.utils.data


    
class Periodic( torch.utils.data.Dataset ):
    
    def fun( self, x ):
        k = (x-self.left)//10 + 1
        retv = numpy.sin( k*x )
        return retv
        
    def fun1( self, x ):
        k = (x-self.left)/10 + 1
        retv = numpy.sin( k*x )
        return retv

    def __init__( self, left, right, dsize, seq=True ):
        self._seq = seq
        self._left = left
        self._right = right
        self._dsize = dsize
        self.x = numpy.linspace( self._left, self._right, self._dsize )
        self.y = self.fun( self.x )

    @property
    def left( self ): return self._left

    @property
    def right( self ): return self._right

    @property
    def range( self ): return [self._left, self._right]

    @property
    def dsize( self ): return self._dsize

    def __len__(self):
        return len( self.x )

    def __getitem__( self, idx ):
        if self._seq:
            return numpy.array( [self.x[idx]] ), self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    # If there is an underlying variable, proving to plotting
    # the underlying values that produce the observed inputs/outputs.
    # The learning algorithms must never access this.
    def plotting_helper( self, n=128 ):
        all_in = numpy.linspace( self._left, self._right, n )
        all_out = self.fun( all_in )
        return None, all_in, all_out

    # Estimate output scale
    @property
    def output_scale( self ):
        return numpy.nanmedian( self.y )



class Sequence( torch.utils.data.Dataset ):
    
    def fun( self, tt ):
        if isinstance( tt, collections.abc.Iterable ):
            retv = numpy.array( [t+12 if t>4 else numpy.exp2(t) if t>=0 else 0 for t in tt] )
        else:
            retv = self.fun( [tt] )[0]
        return retv

    '''
    If seq == True, __getitem__ gives a length-one sequence [x]
    else,  __getitem__ gives the value x
    '''
    def __init__( self, dsize, seq=True ):
        self._seq = seq
        self._left_t = 0.0
        self._right_t = 6.0
        self._dsize = dsize
        # We need to stop at self._right_t-1.0 to have space for
        # xout 1.0 time unit to the right 
        self.t = numpy.linspace( self._left_t, self._right_t-1.0, self._dsize )
        self.xin = self.fun( self.t )
        self.xout = self.fun( self.t+1 )
        # Note that the network is trained on fun() values,
        # not the underlying t variable
        self._left = self.fun( self._left_t )
        self._right = self.fun( self._right_t )

    @property
    def left( self ): return self._left

    @property
    def right( self ): return self._right

    @property
    def range( self ): return [self._left, self._right]

    @property
    def dsize( self ): return self._dsize

    def __len__( self ): return self.dsize

    def __getitem__( self, idx ):
        if self._seq:
            seq = numpy.array( [self.xin[idx]] )
            return seq, self.xout[idx]
        else:
            return self.xin[idx], self.xout[idx]

    # If there is an underlying variable, proving to plotting
    # the underlying values that produce the observed inputs/outputs.
    # The learning algorithms must never access this.
    def plotting_helper( self, n=128 ):
        all_t = numpy.linspace( self._left_t, self._right_t-1, n )
        all_in = self.fun( all_t )
        all_out = self.fun( all_t + 1.0 )
        return all_t, all_in, all_out

    # Estimate output scale
    @property
    def output_scale( self ):
        return numpy.nanmedian( self.xout )
