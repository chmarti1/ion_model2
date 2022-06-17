# SparseN
#   A module for providing the SparseN N-dimensional sparse tensor class
#
# Rev 1.1  3/2022
# Removed dangerous defaults and shape calculations from the ThinSparseN
# definition.


# To Do items:
#   Create overloaded basic operations (multiply, add, subtract, negate)
# that allow the application to demand in-place or thin operations.  All
# binary math operators are currently thick; they result in the creation
# of a new tensor (except for the in-place operations).

import time
import multiprocessing as mp
import numpy as np


__version__ = '1.1'




class SparseN:
    """SparseN  an N-dimensional sparse tensor class
    s = SparseN( (N0, N1, ...) )
        OR
    s = SparseN( s )
        OR
    s = SparseN( d )
    
An empty SparseN matrix can be created by defining its size (N0, N1, ...) or
as a copy of an existing SparseN matrix, S, or from a dense array, D.

Once created, indices can be summoned or set as normal
    S[ii, jj, ...] = value
    value = S[ii, jj, ...]
    
Slices are supported.
    S[:,:,2]    # Returns a SliceSparseN instance.
    ss = S[:,:,2]
    ss[0,1] = 14.
    S[0,1,2]    # Returns 14

Meta data on the tensor shape are available in numpy-compatible member data
    S.ndim      # Number of dimensions
    S.shape     # tensor shape tuple (len(shape) == ndim)
    S.size      # total number of tensor elements
    
Data are stored in internal lists.  These lists should never be modified directly
Always use the documented methods for accessing these values.
    S.index     # list of tuples where values can be found
    S.value     # list of the scalar values in each location
    
There are also metadata available unique to the sparse nature of the system
    S.nnz()     # Function returns the number of non-zero elements

SparseN tensors support basic mathematical operaitons
Unary operations with sparse tensors return ThinSparseN instances.  These
do not contain new copies of the original data.
    -S1
    S1.transpose(..)
    
Binary operations with two sparse tensors:
    S1 + S2
    S1 += S2
    S1 - S2
    S1 -= S2
    S1 * S2         # Inner dot product
    S1.dot(.. S2)   # Multiplicaiton along one or more simultaneous dimensions

Binary operations with dense tensors:
    S+D, D+S    # Returns a dense array
    S += D      # Modifies existing sparse in place
    S-D, D-S    # Returns a dense array
    S -= D
    S*D, D*S    # Inner dot product
    
Binary operations with scalars:
    S+a, a+S
    S += a
    S-a, a-S
    S -= a
    S*a, a*S
"""

    def __init__(self, S):
        # Establish a special case for SliceSparseN copies.
        if isinstance(S, SliceSparseN):
            SparseN.__init__(self, S.shape)
            # Using the iterator will automatically skip the None entries
            # found in SliceSparseN entries.  This is slower than using
            # a list's copy() method, so it is reserved for Slices.
            for index,value in S:
                self.index.append(index)
                self.value.append(value)
        # It is safe to copy for SparseN and ThinSparseN types
        elif isinstance(S, SparseN):
            SparseN.__init__(self, S.shape)
            for index,value in S:
                self.index.append(index)
                self.value.append(value)
        elif isinstance(S, np.ndarray):
            SparseN.__init__(self, S.shape)
            for index,value in zip(np.ndindex(S.shape),np.nditer(S)):
                if value != 0:
                    self.index.append(index)
                    self.value.append(np.asscalar(value))
        else:
            self.shape = tuple(S)
            self.size = np.product(self.shape)
            self.ndim = len(self.shape)
            for dd in self.shape:
                if dd<=0:
                    raise Exception('Tensor dimensions must be positive.')
            self.index = []
            self.value = []
    
    def __setitem__(self, index, value):
        # Check for slicing
        isslice = False
        for ii in index:
            if isinstance(ii, slice):
                isslice = True
                break
                
        # If this is a slice assignment
        if isslice:
            if len(index) != self.ndim:
                raise Exception("GETITEM: number of indices must match the tensor dimension")
            
            S = SliceSparseN(self, index)
            # Case out types of sparse assignments
            # Scalar source
            if np.isscalar(value):
                # No need to zero since we're going to write to all 
                # elements anyway
                for index in np.ndindex(S.shape):
                    S._insert(index, value)
            # Sparse source
            elif isinstance(value,SparseN):
                if S.shape != value.shape:
                    raise Exception('Cannot assign a tensor of shape {:s} to a slice of shape {:s}'.format(repr(value.shape), repr(S.shape)))
                # If the source and destination share parentage, we need to copy the data before moving it
                if self.getparent() is value.getparent():
                    value = SparseN(value)
                # We will need to empty the slice before writing
                S._zero()
                for vi,vv in value:
                    S._insert(vi,vv)
            # Dense array source
            else:
                value = np.asarray(value)
                if S.shape != value.shape:
                    raise Exception('Cannot assign a tensor of shape {:s} to a slice of shape {:s}'.format(repr(value.shape), repr(S.shape)))
                # There is no need to empty the slice since we're going to 
                # write to all the elements anyway.
                for ii,vv in zip(np.ndindex(value.shape), np.nditer(value)):
                    S._insert(ii,np.asscalar(vv))
                    
        # If this is a scalar assignment
        else:
            self._inrange(index)
            self._insert(index, value)
    
    def __delitem__(self, index):
        # Check for slicing
        isslice = False
        for ii in index:
            if isinstance(ii, slice):
                isslice = True
                break
                
        # If this is a slice deletion
        if isslice:
            if len(index) != self.ndim:
                raise Exception("GETITEM: number of indices must match the tensor dimension")
            
            S = SliceSparseN(self, index)
            for si in S.index:
                found, ii = S._find_index(si)
                del self.index[ii]
                del self.value[ii]
                    
        # If this is a single deletion
        else:
            self._inrange(index)
            found, ii = self._find_index(index)
            if found:
                del self.index[ii]
                del self.value[ii]
    
    def __getitem__(self, index):
        if len(index) != self.ndim:
            raise Exception("GETITEM: number of indices must match the tensor dimension")
        # Detect slices
        for ii in index:
            if isinstance(ii,slice):
                return SliceSparseN(self, index)
    
        self._inrange(index)
        found, ii = self._find_index(index)
        if not found:
            return 0.
        return self.value[ii]

        
    def __bool__(self):
        return bool(self.value)
        
    def __len__(self):
        return self.size
        
    def __add__(self, b):
        # If b is dense, use dense arithmatic
        if isinstance(b, (np.ndarray, int, float)):
            c = self.todense()
            np.add(c,b,out=c)
            return c
        c = self.copy()
        c.__iadd__(b)
        return c
        
    def __radd__(self,b):
        return self.__add__(self,b)

    def __iadd__(self, b):
        # Sparse
        if issubclass(type(b), SparseN):
            if b.shape != self.shape:
                raise Exception('Addition and subtraction of tensors requires identical shapes.')
            # If the target and source share parentage, we'll need to make
            # a copy before performing the operation
            if self.getparent() is b.getparent():
                b = SparseN(b)
            for bi, bvalue in b:
                self._increment(bi,bvalue)
        # Dense
        elif isinstance(b, np.ndarray):
            if b.shape != self.shape:
                raise Exception('Addition and subtraction of tensors requires identical shapes.')
            for ii,bb in zip(np.ndindex(b.shape), np.nditer(b)):
                self._increment(ii,bb)
        # Scalar
        else:
            # The increment method looses knowledge of its location in
            # the array after successive calls, so there are opportunities
            # to improve performance here.
            for index in np.ndindex(self.shape):
                self._increment(index, b)
        return self
        
    def __repr__(self):
        out = '<'
        for ii in self.shape[:-1]:
            out += str(ii) + 'x'
        out += str(self.shape[-1]) + ' SparseN tensor with ' + str(self.nnz()) + ' stored elements>'
        return out
        
    def __neg__(self):
        vmap = lambda vv: -vv
        return ThinSparseN(self, vmap=vmap, vmapi = vmap)
        
    def __sub__(self, b):
        if isinstance(b, (np.ndarray, int, float)):
            c = self.todense()
            np.subtract(c,b,out=c)
            return c
        c = self.copy()
        c.__isub__(b)
        return c
        
    def __rsub__(self, b):
        if isinstance(b, (np.ndarray, int, float)):
            c = self.todense()
            np.subtract(b,c,out=c)
            return c
        return self.__neg__().__add__(b)
        
    def __isub__(self, b):
        # Sparse
        if issubclass(type(b), SparseN):
            if b.shape != self.shape:
                raise Exception('Addition and subtraction of tensors requires identical shapes.')
            # If the target and source share parentage, we'll need to make
            # a copy before performing the operation
            if self.getparent() is b.getparent():
                b = SparseN(b)
            for bi, bvalue in b:
                self._increment(bi,-bvalue)
        # Dense
        elif isinstance(b, np.ndarray):
            if b.shape != self.shape:
                raise Exception('Addition and subtraction of tensors requires identical shapes.')
            for ii,bb in zip(np.ndindex(b.shape), np.nditer(b)):
                self._increment(ii,-bb)
        # Scalar
        else:
            # The increment method looses knowledge of its location in
            # the array after successive calls, so there are opportunities
            # to improve performance here.
            for index in np.ndindex(self.shape):
                self._increment(index, -b)
        return self
        
    def __mul__(self, b):
        # Vector/matrix/tensor products
        if isinstance(b, SparseN) or isinstance(b, np.ndarray):
            return self.dot(self.ndim-1, 0, b)
        # Create a thin wrapper for scalar multiplication
        else:
            c = SparseN(self.shape)
            c.index = self.index.copy()
            c.value = [b*vv for vv in self.value]
            return c
        
    def __rmul__(self, b):
        if issubclass(type(b), SparseN) or isinstance(b, np.ndarray):
            return self.dot(0, b.ndim-1, b)
        else:
            return self.__mul__(b)
        
    def __iter__(self):
        return SparseIter(self)
        
    def _inrange(self, index):
        """Tests the index tuple for the correct dimension and range"""
        if len(index) != self.ndim:
            raise Exception('SparseN tensor has %d dimensions, and requires the same number of indices.'%self.ndim)
        for ii, ss in zip(index,self.shape):
            if isinstance(ii,slice):
                pass
            elif isinstance(ii,int):
                if ii < 0 or ii >= ss:
                    raise Exception('Index is out of range: %d'%index)
            else:
                raise Exception('Index is neither an integer or slice: {:s}'.format(repr(ii)))
    
    def _find_index(self, index, iimin=None, iimax=None):
        """Perform a bisection search on the index array
    found, ii = s._find_index( index )
    
index is a tuple of N indices
ii is the integer index in the s.index array where the specified index was 
    found OR where it would be inserted were it to be added.
found is a boolean indicating whether the specified index was located.

To specify a smaller subset of the index list to search, the optional iimin
and iimax parameters may be set.

For speed, very little error checking is done.  _find_index is intended to be
a back-end method, so be careful when using iimin or iimax!  Don't specify 
iimin larger than iimax, and make sure you ACTUALLY include a portion of the
list where the target resides.
"""
        if iimin is None:
            aa = 0
        else:
            aa = iimin

        if iimax is not None:
            bb = iimax
        else:
            bb = len(self.index)-1
            
        # Check to see if the index is even in the range
        if bb < aa:
            return (False, aa)
        elif index <= self.index[aa]:
            return (index == self.index[aa], aa)
        elif index == self.index[bb]:
            return (True, bb)
        elif index > self.index[bb]:
            return (False, bb+1)
            
        # the value definitely lies inside the list, and it is neither aa
        # nor bb.
        while bb-aa>1:
            ii = (aa+bb)//2
            # Eventually, we'll hit the value
            if index == self.index[ii]:
                return (True, ii) 
            elif index < self.index[ii]:
                bb = ii
            else:
                aa = ii
        # Unless the value isn't in the list.
        return (False, bb)

    def _increment(self, index, value):
        """s._increment(index, value)
    
Increments an element of the matrix.  This is equivalent to
    s[index] += value
but it only searches for the index once instead of twice through a call
to __getitem__ and to __setitem__.  The __iadd__, __isub__, __add__, and
__sub__ methods all use this as a tool to perform element-wise 
operations efficiently.
"""
        if value==0:
            return
        found,ii = self._find_index(index)
        if found:
            self.value[ii] += value
            if self.value[ii] == 0:
                del self.index[ii]
                del self.value[ii]
        else:
            self.index.insert(ii, index)
            self.value.insert(ii, value)
            
    def _insert(self, index, value):
        """s._insert(index, value)
        
Places a value in the tensor.  If one already existed, it will be 
overwritten.  If not, the lists will be updated appropriately.
"""
        found, ii = self._find_index(index)
        if found:
            self.value[ii] = value
        else:
            self.index.insert(ii, index)
            self.value.insert(ii, value)
            
    def _zero(self):
        """s._zero()
        
Removes all entries while ignoring the None values returned by slice
instances.  This is equivalent to zeroing a tensor."""
        self.index = []
        self.value = []

    def getparent(self):
        """Returns the SparseN parent that holds the master data
getparent recurses into nested ThinSparseN or SliceSparseN structures"""
        if hasattr(self, "parent"):
            return self.parent.getparent()
        return self

    def nnz(self):
        """Number of non-zero elements"""
        return len(self.value)
        
    def copy(self):
        """another_s = s.copy()
    Makes a new object with copies of the data in the original object.
"""
        return SparseN(self)
        
    def get(self, *index):
        """GET  returns a value in the tensor
    S.get(ii, jj, kk, ... )
    
Returns a scalar in the same way as __getitem__ except that slices are
not supported.  This makes get slightly faster for uses in repeated 
calls to unpatterned indices.  Otherwise, it is generally more efficient
to use slices if possible.

See also: set()
"""
        self._inrange(index)
        found, ii = self._find_index(index)
        if not found:
            return 0.
        return self.value[ii]
        
        
    def set(self, value, *index):
        """SET   assign a scalar value to a tensor element
    S.set(value, ii, jj, kk, ... )

Assigns a scalar value to an element of the tensor in the same way as 
__setitem__ except that slices are not supported.  This makes set() 
slighly faster for uses in repeated calls.

See also: get()
"""
        self._inrange(index)
        self._insert(index,value)
            
            
    def todense(self):
        """Return a dense copy of the tensor
    D = S.todense()
    
Be careful; large multi-dimensional tensors can require large amounts
of memory in dense realizations."""
        d = np.zeros(self.shape)
        for index,value in self:
            d[index] = value
        return d

    def dot(self, adim, bdim, b):
        """Multiply two sparse tensors along the specified dimensions
    C = A.dot(adim, bdim, B)
    
When A is an NA dimensional sparse tensor, and B is an NB dimensional sparse 
tensor or dense array, and there are NN dimensions specified in adims and bdims,
C will be an (NA + NB - 2) dimensional tensor.

For example, if A and B are 3-dimensional, then 
    C = A.dot(2, 0, B)
is equivalent to
    C[i,j,l,m] = sum_k A[i,j,k] * B[k,l,m]
    
The dims may be specified as integers or, to specify simultaneous 
multiplication across multiple indices, as lists or tuples of integers.
So, for the same tensors in the last example,
    C = A.dot(adim=(1,2), bdim=(0,1), B)
is equivalent to
    C[i,l] = sum_j sum_k A[i,j,k] * B[j,k,l]

The order of the dimensions is also important.
    C = A.dot(adim=(1,2), bdim=(1,0), B)
is equivalent to
    C[i,l] = sum_j sum_k A[i,j,k] * B[k,j,l]
    
To force C to be a dense array, use the optional ASDENSE keywrod

    C = A.dot(adim, bdim, B, asdense=True)
"""
        if hasattr(adim,'__iter__') or hasattr(bdim,'__iter__'):
            adim = tuple(adim)
            bdim = tuple(bdim)
            if len(adim) != len(bdim):
                raise Exception('DOT: adim and bdim must specify the same number of dimensions.')
        else:
            # Force a single-element tuple
            adim = (int(adim),)
            bdim = (int(bdim),)
            
        # Check for matching index lengths
        if len(adim) != len(bdim):
            raise Exception('DOT: adim and bdim must have the same length.')
        
        # Check that the dim indices are in-range and that the dimensions match
        for ai in adim:
            if ai < 0 or ai >= self.ndim:
                raise Exception('DOT: Dimension on A not valid. Specified {:d}, only {:d} available.'.format(ai, len(self.shape)))
        for bi in bdim:
            if bi < 0 or bi >= b.ndim:
                raise Exception('DOT: Dimension on B not valid. Specified {:d}, only {:d} available.'.format(bi, len(b.shape)))
        for ai,bi in zip(adim,bdim):
            if self.shape[ai] != b.shape[bi]:
                raise Exception('DOT: Dimensions do not match A.shape[{:d}] = {:d}, B.shape[{:d}] = {:d}.'.format(ai,self.shape[ai],bi,b.shape[bi]))

        # Build lists of the dimensions NOT being multiplied
        # These are maps from the c dimensions to the corresponding a
        # and b dimensions.
        adim_not = []
        for ai in range(self.ndim):
            if ai not in adim:
                adim_not.append(ai)
                
        bdim_not = []
        for bi in range(b.ndim):
            if bi not in bdim:
                bdim_not.append(bi)
        
        # Build the result tensor
        cshape = []
        for ai in adim_not:
            cshape.append(self.shape[ai])
        for bi in bdim_not:
            cshape.append(b.shape[bi])
            
        # Initialize the result
        c = SparseN(cshape)

        # initialize indexes that will be used to perform the multiplication
        bindex = [0] * b.ndim
        cindex = [0] * c.ndim

        # The b-indices (and corresponding c-indices) that are not part of the
        # inner product will always be full slices.
        ci = len(adim_not)
        for bi in bdim_not:
            bindex[bi] = slice(None)
            cindex[ci] = slice(None)
            ci += 1
        # using slices in Numpy's algorithms
        # Loop over the sparse entries
        for aindex,avalue in zip(self.index, self.value):
            # construct the c- and b-indices
            for ai,bi in zip(adim,bdim):
                bindex[bi] = aindex[ai]
            ci = 0
            for ai in adim_not:
                cindex[ci] = aindex[ai]
                ci += 1
            # Perform the product and increment the appropriate c-values
            c[tuple(cindex)] += (avalue * b[tuple(bindex)])
                
        # All done
        return c


    def transpose(self, adim, bdim):
        """Transpose two dimensions in place.
    S.transpose(a,b)
    
a and b are dimensions to transpose so that
    S = SparseN((3,3,3))
    ... code assigning values to S ...
    S2 = SparseN(S)     # Copy S
    S2.transpose(1,2)   # Switch the second and third dimensions
    S[i,j,k] == S2[i,k,j]   # True for all i,j,k
"""
     
        def imap(index):
            return index[:imap.adim] + (index[imap.bdim],) + \
                index[imap.adim+1:imap.bdim] + (index[imap.adim],) + \
                index[imap.bdim+1:]
                
        imap.adim = min(adim,bdim)
        imap.bdim = max(bdim,adim)
        
        # Check legality
        if imap.adim < 0 or imap.bdim >= self.ndim:
            raise Exception('adim and/or bdim are/is out of range.')
        elif adim == bdim:
            return self
   
        # For transpososition, the index map is its own inverse, and it can be
        # used to calculate the new shape.
        return ThinSparseN(self, shape=imap(self.shape), imap=imap, imapi=imap)
            


class VMap:
    """This is a wrapper class for the SparseN index and value lists.  
VMap simulates the behavior of a list, but it applies mapping and 
inverse mapping functions to the values stored there before working with
the values.  In this way, ThinSparseN can use VMap instances to mascarade
as normal index and value lists for the rest of the built-in SparseN
methods.

    VM = VMap(vmap, vmapi, target)
    
    
VMAP
    This is a callable value map function.  It maps values or indicies 
    passed to a ThinSparseN object into the original values or indices.  
    It must be of the form: 
        sparse_value = vmap(thin_sparse_value)

VMAPI
    This is a callable value map inverse function.  It maps values or 
    indices stored in the original SparseN object into the new values or
    indices found in the ThinSparseN object.  It must be of the form: 
        thin_sparse_value = vmapi(sparse_value)

TARGET
    This is the source list of values of indices.
"""
    def __init__(self, vmap, vmapi, target):
        self.vmap = vmap
        self.vmapi = vmapi
        self.target = target
        
    def __getitem__(self, index):
        return self.vmapi(self.target.__getitem__(index))
        
    def __setitem__(self, index, value):
        self.target.__setitem__(index, self.vmap(value))
        
    def __delitem__(self, index):
        self.target.__delitem__(index)
        
    def __iter__(self):
        return map(self.vmapi, self.target)
        
    def __len__(self):
        return len(self.target)
        
    def insert(self, index, value):
        self.target.insert(index, self.vmap(value))
        
    def copy(self):
        """Make an explicit (static) copy of the list.  
This will be a list and NOT a VMAP opject.

    L = vmi.copy()
    """
        return [ii for ii in self]
    

class SliceSparseN(SparseN):
    """A SliceSparseN contains no data, but forms a wrapper around a SparseN.
The intent is to provide a thin wrapper that provides a slice mapping 
into and out-of a SparseN tensor.  SliceSparseN is dissimilar from the
ThinSparseN, because it can change the number of nonzero elements 
available, but it does not provide a means to map or scale the values.

    SS = SliceSparseN(parent, index)
    
PARENT is the target SparseN being sliced.

INDEX is the tuple of indices and slices as would be passed to a 
__getitem__ call.
"""
    def __init__(self, parent, index):
        self.parent = parent
        # check the indices for slices and scale them appropriately
        # iset is the index and slice tuple used to construct this slice
        self.iset = list(index)
        # imap is a list of integers mapping the dimensions of this slice
        # to the matching dimensions of the parent tensor
        self.imap = []
        # Shape is the reduced tensor shape
        self.shape = []
        for dim,ii in enumerate(self.iset):
            if isinstance(ii,slice):
                I = ii.indices(self.parent.shape[dim])
                self.iset[dim] = I
                self.imap.append(dim)
                if I[1]>I[0]:
                    self.shape.append((I[1] - I[0] - 1)//I[2]+1)
                else:
                    self.shape.append(0)
        self.shape = tuple(self.shape)
        # Write the normal SparseN interface members
        self.ndim = len(self.shape)
        self.size = np.product(self.shape)
        self.index = VMap(self._imap, self._imapi, parent.index)
        self.value = parent.value
        # NOTE!!!
        # This approach is a little dangerous, because it makes both the index
        # and value arrays appear to have all the same elements as the parent
        # BUT indices that do not belong to the slice will be returned as
        # None.  The vast majority of methods in the SparseN class use 
        # _find_index, so overloading that definition fixes most issues.
        # However, some still need to iterate over all the elements. For that
        # reason, the SparseIter class is written to be robust against None
        # values in the index - it just skips over them, so the application
        # is blind to the issue.
        # 
        # The virtue is that no redundant lists need to be maintained.  Instead,
        # it is only important that SparseN algorithms be written sensitive to
        # the problem.  __add__ and __sub__ for example should ONLY access 
        # values and indices directly if they can be garanteed to be in a 
        # SparseN and not a SliceSparseN instance.
    
    def _find_index(self, index, iimin=None, iimax=None):
        return self.parent._find_index(self._imap(index), iimin=iimin, iimax=iimax)
        
    def _zero(self):
        """s._zero()
        
Removes all entries while ignoring the None values returned by slice
instances.  This is equivalent to zeroing a tensor."""
        ii = 0
        while ii < len(self.index):
            if self.index[ii] is None:
                ii += 1
            else:
                del self.index[ii]
                del self.value[ii]
        
    def _match(self, index):
        """Test a parent index for membership in the slice"""
        for ii,adj in zip(index, self.iset):
            if isinstance(adj, tuple):
                if ii >= adj[1] or ii < adj[0] or (ii-adj[0])%adj[2]:
                    return False
            elif ii != adj:
                return False
        return True
        
    def _imap(self, index):
        """Map a slice index to its equivalent parent index
        
Since slices always contain a subset of their parents, this should always 
return a valid value.
"""
        parent_index = list(self.iset)
        # Adjust the index to the parent dimensions
        for dim,ii in enumerate(index):
            # imap maps each of the slice dimensions to their original dimension in the parent
            imap = self.imap[dim]
            # adj is the three-element tuple constructed from the slice
            adj = self.iset[imap]
            # Finally, adjust the index to its value in the parent
            parent_index[imap] = adj[0] + adj[2]*ii
        return tuple(parent_index)
        
    def _imapi(self, parent_index):
        """Map a parent index back to the slice index. Return None if it is not a member.
        
Since slices contain a subset of their parents, not all parent entries will
appear in the slice.  When these indices are passed to _imapi, None is 
returned."""
        if not self._match(parent_index):
            return None
        index = []
        for dim,pdim in enumerate(self.imap):
            adj = self.iset[pdim]
            ii = parent_index[pdim]
            index.append( (ii-adj[0])//adj[2] )
        return tuple(index)
        
    def nnz(self):
        """Number of non-zero elements"""
        out = 0
        for index in self.parent.index:
            out += self._match(index)
        return out
      
        

class ThinSparseN(SparseN):
    """A ThinSparseN contains no data, but forms a wrapper around a SparseN.
The intent is that a ThinSparseN will map indices to a parent SparseN
tensor with identical data, but in different arrangements.  This allows
a SparseN to be efficiently resized, transposed, or reshaped.  Write
operations to the child will affect the parent and vice versa.

    TS = ThinSparseN(parent, shape, imap, imapi, vmap, vmapi)

The parent SparseN is required, but the three optional keyword arguments
define the mapping.  The mapping MAY NOT change the number of nonzero 
elements.  It MAY change the tensor's shape and/or dimension.

SHAPE is a tuple defining the shape of the new tensor.  If it is not 
explicitly specified, it will be inherited from the parent SparseN.  If there is
and imap/imapi defined, then shape = imapi(S.shape) will be used.

IMAP, IMAPI are funcitons defining an index maping and inverse mapping.
    TS[ index ] == S[ imap(index) ]
    TS[ imapi(index) ] == S[ index ]
If IMAP is set, but IMAPI is not, then IMAP will be used for both.  This is 
useful in transpose operations, but it will not usually lead to a funcitonal
object.  User beware!

Just like IMAP and IMAPI are funcitons mapping indices for the thin sparse, the
VMAP and VMAPI funcitons are used to map values.
    TS[index] == vmap( S[index] )
    vmapi( TS[index] ) == S[index]
If VMAP is set and VMAPI is not, then VMAP will be used for both.  This is useful
for negation or inverse operations, but it will not usually result in a 
funcitonal object.  User beware!
"""
    def __init__(self, S, shape=None, imap=None, imapi=None, vmap=None, vmapi=None):
        if isinstance(S, ThinSparseN):
            raise exception('Layered ThinSparseN are not currently allowed. ThinSparseN must be built from SparseN tensors.')
        elif not isinstance(S, SparseN):
            raise Exception('The ThinSparseN must be initialized from a SparseN')
        
        # Who is the parent?
        self.parent = S

        # If shape is explicitly defined, use that.  Otherwise, inherit shape from the parent.
        if shape:
            SparseN.__init__(self, shape)
        else:
            SparseN.__init__(self, S.shape)
        
        # This MUST come last because it overwrites value and index
        # Test for index maps
        if imap and imapi:
            self.index = VMap(imap, imapi, S.index)
            self._find_index = self._find_index_wrapper
        elif imap or imapi:
            raise Exception("ThinSparseN: Both IMAP and IMAPI must be specified.")
        else:
            self.index = S.index
        
        # Test for value maps
        if vmap and vmapi:
            # Note that the mapping is reversed from the indices
            self.value = VMap(vmapi, vmap, S.value)
        elif vmap or vmapi:
            raise Exception("ThinSparseN: Both VMAP and VMAPI must be specified.")
        else:
            self.value = S.value

    def _find_index_wrapper(self, index, iimin=None, iimax=None):
        return self.parent._find_index(self.index.vmap(index), iimin=iimin, iimax=iimax)
    
    def _isparent(self, target):
        """Returns True if the target is a parent of self
    _isparent recurses into sparse or thin parents to discover parentage.
"""
        if target is self:
            return True
        return self.parent._isparent(target)
    
    
class SparseIter:
    """An iterator class for sparse tensors

for index,value in SparseIter(S):
    ...
    
Returns the index and values like using zip(), but this iterator skips
the None index entries encountered in SliceSparseN instances.
"""
    def __init__(self, target):
        self.index = target.index.__iter__()
        self.value = target.value.__iter__()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        index = None
        value = None
        while index is None:
            index = self.index.__next__()
            value = self.value.__next__()
        return (index,value)
