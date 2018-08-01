import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from HiggsAnalysis.CombinedLimit.h5pyutils import validateChunkSize, getGrid
from HiggsAnalysis.CombinedLimit.tfsparseutils import SimpleSparseTensor

def maketensor(h5dset):
  
  #special handling for empty arrays
  nelems = 1
  for s in h5dset.shape:
    nelems *= s
  
  esize = np.dtype(h5dset.dtype).itemsize
  print([nelems*esize, h5dset.shape, h5dset.dtype])
  
  if nelems == 0:
    return tf.zeros(h5dset.shape,h5dset.dtype)
  
  #check that chunk shape is compatible with optimized reading strategy
  validateChunkSize(h5dset)
  
  grid = getGrid(h5dset)

  def readChunk(i):
    gv = grid[i]
    readslices = []
    for g,c in zip(gv,h5dset.chunks):
      readslices.append(slice(g,g+c))
    readslices = tuple(readslices)
    #read data from exactly one complete chunk
    aout = h5dset[readslices]
    return aout
      
  #calculate number of chunks
  nchunks = len(grid)  
  
  #create tf Dataset which reads one chunk at a time, unbatches them along the last dimension and  batches them all together,
  #reshapes to restore the original shape, then caches the full result and returns it in an endless loop
  #(unbatching and rebatching is needed because batching only works with equally sized elements, which cannot be guaranteed in case
  #the dataset is not an exact multiple of the chunk size)
  #There are assumptions here about the chunk shape which are enforced by validateChunkSize
  
  dset = tf.data.Dataset.range(nchunks)
  
  dset = dset.map(lambda x: tf.py_func(readChunk,[x],tf.as_dtype(h5dset.dtype)))
  
  #batching not needed in case the whole array is contained in one chunk
  if not h5dset.chunks == h5dset.shape:    
    #calculate largest minibatchsize which evenly divides into dataset
    #(check only multiples of the shape rather than a full prime factorization)"
    minibatchsize = 1
    for s,c in zip(reversed(h5dset.shape),reversed(h5dset.chunks)):
      if c==s:
        minibatchsize *= s
      else:
        break
      
    nbatch = int(h5dset.size/minibatchsize)
    dset = dset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.reshape(x,[-1,minibatchsize])))
    dset = dset.apply(tf.contrib.data.map_and_batch(lambda x: x, nbatch))
    dset = dset.map(lambda x: tf.reshape(x,h5dset.shape))
    
  dset = dset.cache().repeat(-1)
  atensor = dset.make_one_shot_iterator().get_next()
  atensor.set_shape(h5dset.shape)
  return atensor

def makesparsetensor(h5group):
  indices = maketensor(h5group['indices'])
  values = maketensor(h5group['values'])
  dense_shape = h5group['dense_shape'][...]

  return SimpleSparseTensor(indices,values,dense_shape)
  #return tf.SparseTensor(indices,values,dense_shape)
