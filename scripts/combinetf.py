#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser
import multiprocessing

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

import numpy as np
import h5py
import h5py_cache
from HiggsAnalysis.CombinedLimit.tfh5pyutils import maketensor,makesparsetensor
from HiggsAnalysis.CombinedLimit.tfsparseutils import simple_sparse_tensor_dense_matmul, simple_sparse_slice0begin, simple_sparse_to_dense, SimpleSparseTensor
from HiggsAnalysis.CombinedLimit.lsr1trustobs import SR1TrustExact
import scipy
import math
import time


# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

#from root_numpy import array2hist

from array import array

from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface,jacobian,sum_loop
from tensorflow.python.ops.parallel_for.gradients import jacobian as pjacobian
from tensorflow.python.ops.parallel_for.control_flow_ops import for_loop

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
parser.add_option("-t","--toys", default=0, type=int, help="run a given number of toys, 0 fits the data (default), and -1 fits the asimov toy")
parser.add_option("","--toysFrequentist", default=True, action='store_true', help="run frequentist-type toys by randomizing constraint minima")
parser.add_option("","--bypassFrequentistFit", default=True, action='store_true', help="bypass fit to data when running frequentist toys to get toys based on prefit expectations")
parser.add_option("","--bootstrapData", default=False, action='store_true', help="throw toys directly from observed data counts rather than expectation from templates")
parser.add_option("","--randomizeStart", default=False, action='store_true', help="randomize starting values for fit (only implemented for asimov dataset for now")
parser.add_option("","--tolerance", default=1e-3, type=float, help="convergence tolerance for minimizer")
parser.add_option("","--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_option("","--seed", default=123456789, type=int, help="random seed for toys")
parser.add_option("","--fitverbose", default=0, type=int, help="verbosity level for fit")
parser.add_option("","--minos", default=[], type="string", action="append", help="run minos on the specified variables")
parser.add_option("","--scan", default=[], type="string", action="append", help="run likelihood scan on the specified variables")
parser.add_option("","--scanPoints", default=16, type=int, help="default number of points for likelihood scan")
parser.add_option("","--scanRange", default=3., type=float, help="default scan range in terms of hessian uncertainty")
parser.add_option("","--nThreads", default=-1., type=int, help="set number of threads (default is -1: use all available cores)")
parser.add_option("","--POIMode", default="mu",type="string", help="mode for POI's")
parser.add_option("","--nonNegativePOI", default=True, action='store_true', help="force signal strengths to be non-negative")
parser.add_option("","--POIDefault", default=1., type=float, help="mode for POI's")
parser.add_option("","--doBenchmark", default=False, action='store_true', help="run benchmarks")
parser.add_option("","--saveHists", default=False, action='store_true', help="run benchmarks")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)
    
seed = options.seed
print(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

options.fileName = args[0]

cacheSize = 4*1024**2
#TODO open file an extra time and enforce sufficient cache size for second file open
f = h5py_cache.File(options.fileName, chunk_cache_mem_size=cacheSize, mode='r')

#load text arrays from file
procs = f['hprocs'][...]
signals = f['hsignals'][...]
systs = f['hsysts'][...]
maskedchans = f['hmaskedchans'][...]

#load arrays from file
hdata_obs = f['hdata_obs']

sparse = not 'hnorm' in f

if sparse:
  hnorm_sparse = f['hnorm_sparse']
  hlogk_sparse = f['hlogk_sparse']
  nbinsfull = hnorm_sparse.attrs['dense_shape'][0]
else:  
  hnorm = f['hnorm']
  hlogk = f['hlogk']
  nbinsfull = hnorm.attrs['original_shape'][0]

#infer some metadata from loaded information
dtype = hdata_obs.dtype
nbins = hdata_obs.shape[-1]
nbinsmasked = nbinsfull - nbins
nproc = len(procs)
nsyst = len(systs)
nsignals = len(signals)

#build tensorflow graph for likelihood calculation

#start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
#note that this does NOT trigger the actual reading from disk, since this only happens when the
#returned tensors are evaluated for the first time inside the graph
data_obs = maketensor(hdata_obs)
if sparse:
  norm_sparse = makesparsetensor(hnorm_sparse)
  logk_sparse = makesparsetensor(hlogk_sparse)
else:
  norm = maketensor(hnorm)
  logk = maketensor(hlogk)

if options.nonNegativePOI:
  boundmode = 1
else:
  boundmode = 0

pois = []  
  
if options.POIMode == "mu":
  npoi = nsignals
  poidefault = options.POIDefault*tf.ones([npoi],dtype=dtype)
  for signal in signals:
    pois.append(signal)
elif options.POIMode == "none":
  npoi = 0
  poidefault = tf.zeros([],dtype=dtype)
else:
  raise Exception("unsupported POIMode")

nparms = npoi + nsyst
parms = np.concatenate([pois,systs])

if boundmode==0:
  xpoidefault = poidefault
elif boundmode==1:
  xpoidefault = tf.sqrt(poidefault)

print("nbins = %d, nbinsfull = %d, nproc = %d, npoi = %d, nsyst = %d" % (nbins,nbinsfull,nproc, npoi, nsyst))

#data
nobs = tf.Variable(data_obs, trainable=False, name="nobs")
theta0 = tf.Variable(tf.zeros([nsyst],dtype=dtype), trainable=False, name="theta0")

#tf variable containing all fit parameters
thetadefault = tf.zeros([nsyst],dtype=dtype)
if npoi>0:
  xdefault = tf.concat([xpoidefault,thetadefault], axis=0)
else:
  xdefault = thetadefault
  
x = tf.Variable(xdefault, name="x")

xpoi = x[:npoi]
theta = x[npoi:]

if boundmode == 0:
  poi = xpoi
  gradr = tf.ones_like(poi)
elif boundmode == 1:
  poi = tf.square(xpoi)
  gradr = 2.*xpoi

#vector encoding effect of signal strengths
if options.POIMode == "mu":
  r = poi
elif options.POIMode == "none":
  r = tf.ones([nsignals],dtype=dtype)

rnorm = tf.concat([r,tf.ones([nproc-nsignals],dtype=dtype)],axis=0)
mrnorm = tf.expand_dims(rnorm,-1)
ernorm = tf.reshape(rnorm,[1,-1])

#interpolation for asymmetric log-normal
twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3.*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)

thetaalpha = theta*alpha

mthetaalpha = tf.stack([theta,thetaalpha],axis=0) #now has shape [2,nsyst]
mthetaalpha = tf.reshape(mthetaalpha,[2*nsyst,1])

gradtheta = tf.ones_like(theta)
#TODO implement real gradient
gradthetaalpha = tf.zeros_like(theta)
gradethetaalpha = tf.stack([gradtheta,gradthetaalpha],axis=0)
gradethetaalpha = tf.reshape(gradethetaalpha,[1,1,2,nsyst])

#gradmthetaalpha = tf.stack([gradtheta,gradthetaalpha],axis=0)
#gradmthetaalpha = tf.reshape(gradmthetaalpha,[2*nsyst,1])

gradrnorm = tf.concat([gradr,tf.zeros([nproc-nsignals],dtype=dtype)],axis=0)
gradmrnorm = tf.expand_dims(gradrnorm,-1)
gradernorm = tf.reshape(gradrnorm,[1,-1])


if sparse:  
  logsnorm = simple_sparse_tensor_dense_matmul(logk_sparse,mthetaalpha)
  logsnorm = tf.squeeze(logsnorm,-1)
  snorm = tf.exp(logsnorm)
  
  snormnorm_sparse = SimpleSparseTensor(norm_sparse.indices, snorm*norm_sparse.values, norm_sparse.dense_shape)
  nexpfull = simple_sparse_tensor_dense_matmul(snormnorm_sparse,mrnorm)
  nexpfull = tf.squeeze(nexpfull,-1)

  #slice the sparse tensor along axis 0 only, since this is simpler than slicing in
  #other dimensions due to the ordering of the tensor,
  #after this the result should be relatively small in any case and  further
  #manipulations can be done more efficiently after converting to dense
  snormnormmasked0_sparse = simple_sparse_slice0begin(snormnorm_sparse, nbins, doCache=True)
  snormnormmasked0 = simple_sparse_to_dense(snormnormmasked0_sparse)
  snormnormmasked = snormnormmasked0[:,:nsignals]
  
  #TODO consider doing this one column at a time to save memory
  if options.saveHists:
    snormnorm = simple_sparse_to_dense(snormnorm_sparse)
    
    jacnexpfullr = gradernorm*snormnorm
    jacnexpfullr = jacnexpfullr[:,:nsignals]
    jacnexpsigr = jacnexpfullr
    jacnexpbkgr = tf.zeros_like(jacnexpfullr)
    
    normfull = ernorm*snormnorm
    nexpsig = tf.reduce_sum(normfull[:,:nsignals],axis=-1)
    nexpbkg = tf.reduce_sum(normfull[:,nsignals:],axis=-1)
  
else:
  #matrix encoding effect of nuisance parameters
  #memory efficient version (do summation together with multiplication in a single tensor contraction step)
  #this is equivalent to 
  #alpha = tf.reshape(alpha,[-1,1,1])
  #theta = tf.reshape(theta,[-1,1,1])
  #logk = logkavg + alpha*logkhalfdiff
  #logktheta = theta*logk
  #logsnorm = tf.reduce_sum(logktheta, axis=0)
  
  mlogk = tf.reshape(logk,[nbinsfull*nproc,2*nsyst])
  logsnorm = tf.matmul(mlogk,mthetaalpha)
  logsnorm = tf.reshape(logsnorm,[nbinsfull,nproc])

  snorm = tf.exp(logsnorm)

  #final expected yields per-bin including effect of signal
  #strengths and nuisance parmeters
  #memory efficient version (do summation together with multiplication in a single tensor contraction step)
  #equivalent to (with some reshaping to explicitly match indices)
  #rnorm = tf.reshape(rnorm,[1,-1])
  #pnormfull = rnorm*snorm*norm
  #nexpfull = tf.reduce_sum(pnormfull,axis=-1)
  snormnorm = snorm*norm  
  nexpfull = tf.matmul(snormnorm, mrnorm)
  nexpfull = tf.squeeze(nexpfull,-1)

  snormnormmasked = snormnorm[nbins:,:nsignals]
  
  if options.saveHists:
    
    jacnexpfullr = gradernorm*snormnorm
    jacnexpfullr = jacnexpfullr[:,:nsignals]
    jacnexpsigr = jacnexpfullr
    jacnexpbkgr = tf.zeros_like(jacnexpfullr)
    
    normfull = ernorm*snormnorm
    nexpsig = tf.reduce_sum(normfull[:,:nsignals],axis=-1)
    nexpbkg = tf.reduce_sum(normfull[:,nsignals:],axis=-1)
  
    #TODO, re-express these as matrix multiplications and/or einsum
    enormfull = tf.reshape(normfull,[nbinsfull,nproc,1,1])
    jacnormfulltheta = logk*gradethetaalpha*enormfull
    jacnexpfulltheta = tf.reduce_sum(jacnormfulltheta,axis=[1,2])
    jacnexpsigtheta = tf.reduce_sum(jacnormfulltheta[:,:nsignals],axis=[1,2])
    jacnexpbkgtheta = tf.reduce_sum(jacnormfulltheta[:,nsignals:],axis=[1,2])
    
    jacnexpfull = tf.concat([jacnexpfullr,jacnexpfulltheta],axis=-1)
    jacnexpsig = tf.concat([jacnexpsigr,jacnexpsigtheta],axis=-1)
    jacnexpbkg = tf.concat([jacnexpbkgr,jacnexpbkgtheta],axis=-1)
    
    
pmaskedexp = r*tf.reduce_sum(snormnormmasked,axis=0)

maskedexp = nexpfull[nbins:]

#matrix multiplication below is equivalent to
#pmaskedexpnorm = r*tf.reduce_sum(snormnormmasked/maskedexp, axis=0)

mmaskedexpr = tf.expand_dims(tf.reciprocal(maskedexp),0)
pmaskedexpnorm = tf.matmul(mmaskedexpr,snormnormmasked)
pmaskedexpnorm = tf.squeeze(pmaskedexpnorm,0)
pmaskedexpnorm = r*pmaskedexpnorm
 
nexp = nexpfull[:nbins]

nexpsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexp)
lognexp = tf.log(nexpsafe)

nexpnom = tf.Variable(nexp, trainable=False, name="nexpnom")
nexpnomsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexpnom)
lognexpnom = tf.log(nexpnomsafe)

#final likelihood computation

#poisson term  
lnfull = tf.reduce_sum(-nobs*lognexp + nexp, axis=-1)

#poisson term with offset to improve numerical precision
ln = tf.reduce_sum(-nobs*(lognexp-lognexpnom) + nexp-nexpnom, axis=-1)

#constraints
lc = tf.reduce_sum(0.5*tf.square(theta - theta0))

l = ln + lc
lfull = lnfull + lc
 
#name outputs
poi = tf.identity(poi, name=options.POIMode)
pmaskedexp = tf.identity(pmaskedexp, "pmaskedexp")
pmaskedexpnorm = tf.identity(pmaskedexpnorm, "pmaskedexpnorm")
 
outputs = []

outputs.append(poi)
if nbinsmasked>0:
  outputs.append(pmaskedexp)
  outputs.append(pmaskedexpnorm)

nthreadshess = options.nThreads
if nthreadshess<0:
  nthreadshess = multiprocessing.cpu_count()
nthreadshess = min(nthreadshess,nparms)

grad = tf.gradients(l,x,gate_gradients=True)[0]  
hessian = jacobian(grad,x,gate_gradients=True,parallel_iterations=nthreadshess,back_prop=False)

eigvals = tf.self_adjoint_eigvals(hessian)
mineigv = tf.reduce_min(eigvals)
isposdef = mineigv > 0.
invhessian = tf.matrix_inverse(hessian)
#invhessianchol = tf.transpose(tf.cholesky(invhessian))
invhessianchol = tf.cholesky(invhessian)
gradcol = tf.reshape(grad,[-1,1])
edm = 0.5*tf.matmul(tf.matmul(gradcol,invhessian,transpose_a=True),gradcol)

invhessianouts = []
for output in outputs:
  jacout = jacobian(tf.concat([output,theta],axis=0),x,gate_gradients=True,parallel_iterations=nthreadshess,back_prop=False)
  invhessianout = tf.matmul(jacout,tf.matmul(invhessian,jacout,transpose_b=True))
  invhessianouts.append(invhessianout)

l0 = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
x0 = tf.Variable(np.zeros(x.shape,dtype=dtype),trainable=False)
a = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
errdir = tf.Variable(np.zeros(x.shape,dtype=dtype),trainable=False)
dlconstraint = l - l0

def experrslow(expected, invhess):
  #compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
  
  flatexp = tf.reshape(expected,[-1])
  expsize = tf.shape(flatexp)[0]
  
  arr = tf.TensorArray(expected.dtype,size=expsize,infer_shape=True,element_shape=[])
  loop_vars = [arr, tf.constant(0)]
  def cond(arr,j):
    return (j < expsize)
  
  def body(arr,j):
    j = tf.Print(j,[j])
    #compute the jacobian
    expgrad = tf.gradients(flatexp[j],x,gate_gradients=True)[0]
    expgradcol = tf.reshape(expgrad,[-1,1])
    #this is equivalent to jac cov jac^T
    err = tf.sqrt(tf.reduce_sum(expgradcol*tf.matmul(invhess,expgradcol,b_is_sparse=False)))
    #err = tf.sqrt(tf.reduce_sum(tf.square(expgradcol)))
    arr = arr.write(j,err)
    return (arr,j+1)
  
  arr,j = tf.while_loop(cond,body,loop_vars, parallel_iterations=64, back_prop=False)
  errv = arr.stack()
  errm = tf.reshape(errv,expected.shape)
  return errm

def experr(expected, invhesschol):
  #compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
  
  #flatexp = tf.reshape(expected,[-1])
  #expsize = flatexp.shape[0]
  #u = tf.ones_like(flatexp)
  u = tf.ones_like(expected)
  #eu = tf.exp(u)
  
  #v = tf.ones_like(x)
  #ev = tf.exp(v)
  
  #uin = tf.ones_like(flatexp)
  #vin = tf.ones_like(x)
  
  #m = tf.reshape(uin,[-1,1])*tf.reshape(vin,[1,-1])
  #mflat = tf.reshape(m,[-1])
  #u = tf.reduce_mean(tf.gradients(mflat,vin,gate_gradients=True)[0]
  
  #m = tf.ones([expsize*nparms],dtype=flatexp.dtype)
  #s,u,v = tf.svd(tf.reshape(m,[expsize,nparms]), full_matrices=False)
  #s0 = s[0]
  #u = tf.sqrt(s0)*u[:,0]
  #v = tf.sqrt(s0)*v[:,0]
  
  #u = tf.Print(u,[u],message="u",summarize=10000)
  #v = tf.Print(v,[v],message="v",summarize=10000)
  #v = tf.Print(v,[s],message="s",summarize=10000)

  
  #print("ushape = %r" % u.shape)
  #print("vshape = %r" % v.shape)
  
  #dndx = tf.gradients(flatexp, x, grad_ys = u, gate_gradients=True)[0]
  #jacv = tf.gradients(dndx, m, grad_ys = v, gate_gradients=True)[0]
  #jac = tf.reshape(jacv,[expsize,nparms])
  #jac = tf.Print(jac,[jac],message="jac",summarize=10000)
  #jacprod = tf.matmul(jac,invhesschol,transpose_b=True)
  #errv = tf.sqrt(tf.reduce_sum(tf.square(jacprod),axis=-1))
  
  dndx = tf.gradients(expected, x, grad_ys = u, gate_gradients=True)[0]
  #dndx = tf.gradients(flatexp, x, grad_ys = u, gate_gradients=True)[0]
  #dndxalt = tf.gradients(dndx, u, grad_ys = v, gate_gradients=True)[0]
  #dndxre = tf.gradients(dndxalt, v, gate_gradients=True)[0]
  
  dndxv = tf.reshape(dndx,[-1,1])
  choldndxv = tf.matmul(tf.stop_gradient(invhesschol),dndxv,transpose_a=True)
  choldndx = tf.reshape(choldndxv,[-1])
  #errj = pjacobian(choldndx,u,use_pfor=False)
  #errsq = tf.square(errj)
  #err = tf.sqrt(tf.reduce_sum(errsq,axis=0))
  #return err
  
  #choldndx = dndx
  
  #errsqs = []
  #for j in range(nparms):
    #errsq = tf.square(tf.gradients(choldndx[j], u, gate_gradients=True)[0])
    #errsqs.append(errsq)
    
  #err = tf.sqrt(tf.accumulate_n(errsqs))
  #return err
  
  def forbody(j):
    grad = tf.square(tf.gradients(tf.gather(choldndx,j), u, gate_gradients=True)[0])
    return grad
    
  errj = sum_loop(forbody,tf.zeros_like(expected),nparms,parallel_iterations=nthreadshess, back_prop=False)
  print("errj:")
  print(errj)
  
  #errsq = tf.square(errj)
  #err = tf.sqrt(tf.reduce_sum(errsq,axis=0))
  err = tf.sqrt(errj)
  return err
  
  #arr = tf.TensorArray(expected.dtype,nparms)
  #loop_vars = [arr,tf.constant(0)]
  #def cond(arr,j):
    #return j<nparms
  
  #def body(arr,j):
    ##grad = tf.expand_dims(tf.gradients(tf.gather(choldndx,j),u)[0],0)
    #arr = arr.write(j,forbody(j))
    #return (arr,j+1)
  
  #arr,j = tf.while_loop(cond,body,loop_vars, parallel_iterations=nthreadshess, back_prop=False)
  ##errsq = tf.square(arr.concat())
  #errsq = tf.square(arr.stack())
  #err = tf.sqrt(tf.reduce_sum(errsq,axis=0))
  #return err
  
  ##arr = tf.TensorArray(dtype=expected.dtype, size=nparms, element_shape=expected.shape)
  #arr = tf.TensorArray(dtype=expected.dtype, size=nparms)
  #loop_vars = [arr, tf.constant(0)]
  ##loop_vars = [tf.zeros_like(expected), tf.constant(0)]
  #def cond(errsq,j):
    #return (j < nparms)
  
  #def body(errsq,j):
    ##j = tf.Print(j,[j])
    
    
    ##gradsq = tf.square(tf.gradients(choldndx[j], u, gate_gradients=True)[0])
    ##gradsq = tf.gradients(choldndx[j], u, gate_gradients=True)[0]
    ##gradsq = tf.gradients(tf.gather(choldndx,j), u, gate_gradients=True)[0]
    #gradsq = tf.gradients(tf.gather(choldndx,j), u)[0]
    ##errsqout = errsq + gradsq
    #errsqout = errsq.write(j,tf.expand_dims(gradsq,0))
    
    ##errsq = errsq + tf.square(tf.gradients(choldndx[j], u, stop_gradients=x, gate_gradients=True)[0])
    
    ##errsq = errsq + tf.square(tf.gradients(dndx,flatexp, grad_ys=invhesschol[j], gate_gradients=True)[0])
    
    ##rjacj = tf.gradients(flatexp,x,
    
    ##compute the jacobian
    ##expgrad = tf.gradients(flatexp[j],x,gate_gradients=True)[0]
    ##expgradcol = tf.reshape(expgrad,[-1,1])
    ###this is equivalent to jac cov jac^T
    ##err = tf.sqrt(tf.reduce_sum(expgradcol*tf.matmul(invhess,expgradcol,b_is_sparse=False)))
    ###err = tf.sqrt(tf.reduce_sum(tf.square(expgradcol)))
    ##arr = arr.write(j,err)
    #return (errsqout,j+1)
  
  ##errsqloop,j = tf.while_loop(cond,body,loop_vars, parallel_iterations=nthreadshess, back_prop=False)
  #errsqloop,j = tf.while_loop(cond,body,loop_vars)
  
  ##errsqt = tf.square(errsqloop.stack())
  
  #errsqt = tf.square(errsqloop.concat())
  ##errsqt = tf.reshape(errsqt,expected.shape)
  ##errsql = tf.unstack(errsqt,num=nparms,axis=0)
  ##err = tf.accumulate_n(errsql)
  ##err = tf.sqrt(tf.accumulate_n(errsqloop))
  #err = tf.sqrt(tf.reduce_sum(errsqt,axis=0))
  ##errv = arr.stack()
  #err = tf.sqrt(errsqloop)
  #errv = tf.sqrt(errsqloop)
  #errm = tf.reshape(errv,expected.shape)
  #return err

def experrfast(jac, invhess):
  errm = tf.sqrt(tf.reduce_sum(tf.matmul(jac,invhess)*jac,axis=-1))
  return errm
  
if options.saveHists:
  #for prefit uncertainties assume zero uncertainty on pois since this is not well defined
  #and uncorrelated unit uncertainties on nuisances parameters
  invhessianprefit = tf.diag(tf.concat([tf.zeros_like(xpoi),tf.ones_like(theta)],axis=0))
  
  #compute uncertainties for expectations (prefit)
  normfullerrpre = experr(normfull,invhessianprefit)
  nexpfullerrpre = experr(nexpfull,invhessianprefit)
  nexpsigerrpre = experr(nexpsig, invhessianprefit)
  nexpbkgerrpre = experr(nexpbkg, invhessianprefit)
  
  ##compute uncertainties for expectations (postfit, using the full covariance matrix)
  normfullerr = experr(normfull,invhessianchol)
  nexpfullerr = experr(nexpfull,invhessianchol)
  nexpsigerr = experr(nexpsig, invhessianchol)
  nexpbkgerr = experr(nexpbkg, invhessianchol)
  
  #nexpfullerrpre = tf.sqrt(tf.reduce_sum(tf.matmul(jacnexpfull,invhessianprefit)*jacnexpfull,axis=-1))
  #nexpfullerr = tf.sqrt(tf.reduce_sum(tf.matmul(jacnexpfull,invhessian)*jacnexpfull,axis=-1))
  
  #nexpfullerrpre = experr(jacnexpfull,invhessianprefit)
  #nexpsigerrpre = experr(jacnexpsig,invhessianprefit)
  #nexpbkgerrpre = experr(jacnexpbkg,invhessianprefit)

  #nexpfullerr = experr(jacnexpfull,invhessian)
  #nexpsigerr = experr(jacnexpsig,invhessian)
  #nexpbkgerr = experr(jacnexpbkg,invhessian)
  
  

lb = np.concatenate((-np.inf*np.ones([npoi],dtype=dtype),-np.inf*np.ones([nsyst],dtype=dtype)),axis=0)
ub = np.concatenate((np.inf*np.ones([npoi],dtype=dtype),np.inf*np.ones([nsyst],dtype=dtype)),axis=0)

xtol = np.finfo(dtype).eps
edmtol = math.sqrt(xtol)
btol = 1e-8
tfminimizer = SR1TrustExact(l,x,grad)
opinit = tfminimizer.initialize(l,x,grad,hessian)
opmin = tfminimizer.minimize(l,x,grad)

scanvars = {}
scannames = []
scanvars["x"] = x
scannames.append("x")
for output in outputs:
  outname = ":".join(output.name.split(":")[:-1])
  outputtheta = tf.concat([output,theta],axis=0)
  scanvars[outname] = outputtheta
  scannames.append(outname)

scanminimizers = {}
minosminimizers = {}
for scanname in scannames:
  scanvar = scanvars[scanname]
  errproj = -tf.reduce_sum((scanvar-x0)*errdir,axis=0)
  dxconstraint = a + errproj
  scanminimizer = ScipyTROptimizerInterface(l, var_list = [x], var_to_bounds={x: (lb,ub)},  equalities=[dxconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
  minosminimizer = ScipyTROptimizerInterface(errproj, var_list = [x], var_to_bounds={x: (lb,ub)},  equalities=[dlconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
  scanminimizers[scanname] = scanminimizer
  minosminimizers[scanname] = minosminimizer

globalinit = tf.global_variables_initializer()
nexpnomassign = tf.assign(nexpnom,nexp)
dataobsassign = tf.assign(nobs,data_obs)
asimovassign = tf.assign(nobs,nexp)
asimovrandomizestart = tf.assign(x,tf.clip_by_value(tf.contrib.distributions.MultivariateNormalFullCovariance(x,invhessian).sample(),lb,ub))
bootstrapassign = tf.assign(nobs,tf.random_poisson(nobs,shape=[],dtype=dtype))
toyassign = tf.assign(nobs,tf.random_poisson(nexp,shape=[],dtype=dtype))
frequentistassign = tf.assign(theta0,theta + tf.random_normal(shape=theta.shape,dtype=dtype))
thetastartassign = tf.assign(x, tf.concat([xpoi,theta0],axis=0))
bayesassign = tf.assign(x, tf.concat([xpoi,theta+tf.random_normal(shape=theta.shape,dtype=dtype)],axis=0))

#initialize output tree
f = ROOT.TFile( 'fitresults_%i.root' % seed, 'recreate' )
tree = ROOT.TTree("fitresults", "fitresults")

tseed = array('i', [seed])
tree.Branch('seed',tseed,'seed/I')

titoy = array('i',[0])
tree.Branch('itoy',titoy,'itoy/I')

tstatus = array('i',[0])
tree.Branch('status',tstatus,'status/I')

terrstatus = array('i',[0])
tree.Branch('errstatus',terrstatus,'errstatus/I')

tscanidx = array('i',[0])
tree.Branch('scanidx',tscanidx,'scanidx/I')

tedmval = array('d',[0.])
tree.Branch('edmval',tedmval,'edmval/D')

tnllval = array('d',[0.])
tree.Branch('nllval',tnllval,'nllval/D')

tnllvalfull = array('d',[0.])
tree.Branch('nllvalfull',tnllvalfull,'nllvalfull/D')

tdnllval = array('d',[0.])
tree.Branch('dnllval',tdnllval,'dnllval/D')

tchisq = array('d',[0.])
tree.Branch('chisq', tchisq, 'chisq/D')

tchisqpartial = array('d',[0.])
tree.Branch('chisqpartial', tchisqpartial, 'chisqpartial/D')

tndof = array('i',[0])
tree.Branch('ndof',tndof,'ndof/I')

tndofpartial = array('i',[0])
tree.Branch('ndofpartial',tndofpartial,'ndofpartial/I')

toutvalss = []
touterrss = []
toutminosupss = []
toutminosdownss = []
toutgenvalss = []
outnames = []
outidxs = {}
for iout,output in enumerate(outputs):
  outname = ":".join(output.name.split(":")[:-1])
  outnames.append(outname)
  outidxs[outname] = iout
  
  toutvals = []
  touterrs = []
  toutminosups = []
  toutminosdowns = []
  toutgenvals = []
  
  toutvalss.append(toutvals)
  touterrss.append(touterrs)
  toutminosupss.append(toutminosups)
  toutminosdownss.append(toutminosdowns)
  toutgenvalss.append(toutgenvals)
    
  for poi in pois:
    toutval = array('f', [0.])
    touterr = array('f', [0.])
    toutminosup = array('f', [0.])
    toutminosdown = array('f', [0.])
    toutgenval = array('f', [0.])
    toutvals.append(toutval)
    touterrs.append(touterr)
    toutminosups.append(toutminosup)
    toutminosdowns.append(toutminosdown)
    toutgenvals.append(toutgenval)
    basename = "%s_%s" % (poi,outname)
    tree.Branch(basename, toutval, '%s/F' % basename)
    tree.Branch('%s_err' % basename, touterr, '%s_err/F' % basename)
    tree.Branch('%s_minosup' % basename, toutminosup, '%s_minosup/F' % basename)
    tree.Branch('%s_minosdown' % basename, toutminosdown, '%s_minosdown/F' % basename)
    tree.Branch('%s_gen' % basename, toutgenval, '%s_gen/F' % basename)

tthetavals = []
ttheta0vals = []
tthetaerrs = []
tthetaminosups = []
tthetaminosdowns = []
tthetagenvals = []
for syst in systs:
  systname = syst
  tthetaval = array('f', [0.])
  ttheta0val = array('f', [0.])
  tthetaerr = array('f', [0.])
  tthetaminosup = array('f', [0.])
  tthetaminosdown = array('f', [0.])
  tthetagenval = array('f', [0.])
  tthetavals.append(tthetaval)
  ttheta0vals.append(ttheta0val)
  tthetaerrs.append(tthetaerr)
  tthetaminosups.append(tthetaminosup)
  tthetaminosdowns.append(tthetaminosdown)
  tthetagenvals.append(tthetagenval)
  tree.Branch('%s' % systname, tthetaval, '%s/F' % systname)
  tree.Branch('%s_In' % systname, ttheta0val, '%s_In/F' % systname)
  tree.Branch('%s_err' % systname, tthetaerr, '%s_err/F' % systname)
  tree.Branch('%s_minosup' % systname, tthetaminosup, '%s_minosup/F' % systname)
  tree.Branch('%s_minosdown' % systname, tthetaminosdown, '%s_minosdown/F' % systname)
  tree.Branch('%s_gen' % systname, tthetagenval, '%s_gen/F' % systname)

ntoys = options.toys
if ntoys <= 0:
  ntoys = 1

#initialize tf session
if options.nThreads>0:
  config = tf.ConfigProto(intra_op_parallelism_threads=options.nThreads, inter_op_parallelism_threads=options.nThreads)
else:
  config = None

sess = tf.Session(config=config)

#note that initializing all variables also triggers reading the hdf5 arrays from disk and populating the caches
print("initializing variables (this will trigger loading of large arrays from disk)")
sess.run(globalinit)
for cacheinit in tf.get_collection("cache_initializers"):
  sess.run(cacheinit)

xv = sess.run(x)

#set likelihood offset
sess.run(nexpnomassign)

outvalsgens,thetavalsgen = sess.run([outputs,theta])

#all caches should be filled by now

def minimize():
  sess.run(opinit)
  ifit = 0
  while True:
    isconverged,_ = sess.run(opmin)
    if options.fitverbose > 2:
      lval, gmagval, e0val, trval = sess.run([tfminimizer.loss_old, tfminimizer.grad_old_mag, tfminimizer.e0, tfminimizer.trustradius])
      print('Iteration %d, loss = %.6f, |g| = %e, lowest eigenvalue = %e, trustradius = %e' % (ifit,lval,gmagval,e0val,trval))
    if isconverged:
      break
    
    ifit += 1

def fillHists(tag):
  print("filling hists")
  hists = []
  
  if tag=='prefit':
    normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval, nexpsigerrval, nexpbkgerrval, normfullerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerrpre,nexpsigerrpre,nexpbkgerrpre,normfullerrpre])
    #normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval, nexpsigerrval, nexpbkgerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerrpre,nexpsigerrpre,nexpbkgerrpre])
    #normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerrpre])
  else:
    normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval, nexpsigerrval, nexpbkgerrval, normfullerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerr,nexpsigerr,nexpbkgerr,normfullerr])
    #normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval, nexpsigerrval, nexpbkgerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerr,nexpsigerr,nexpbkgerr])
    #normfullval, nexpfullval, nexpsigval, nexpbkgval, nexpfullerrval = sess.run([normfull,nexpfull,nexpsig,nexpbkg,nexpfullerr])
  
  #expfullHist = ROOT.TH1D('expfull_%s' % tag,'',nbinsfull,-0.5, float(nbinsfull)-0.5)
  #hists.append(expfullHist)
  ##array2hist(nexpfullval,expfullHist, errors=nexpfullerrval)
  
  #expsigHist = ROOT.TH1D('expsig_%s' % tag,'',nbinsfull,-0.5, float(nbinsfull)-0.5)
  #hists.append(expsigHist)
  ##array2hist(nexpsigval,expsigHist, errors=nexpsigerrval)
  
  #expbkgHist = ROOT.TH1D('expbkg_%s' % tag,'',nbinsfull,-0.5, float(nbinsfull)-0.5)
  #hists.append(expbkgHist)
  #array2hist(nexpbkgval,expbkgHist, errors=nexpbkgerrval)
  
  #for iproc,proc in enumerate(procs):
    #expHist = ROOT.TH1D('expproc_%s_%s' % (proc,tag),'',nbinsfull,-0.5, float(nbinsfull)-0.5)
    #hists.append(expHist)
    ##array2hist(normfullval[:,iproc], expHist,errors=normfullerrval[:,iproc])
    ##array2hist(normfullval[:,iproc], expHist)
  
  print("done filling hists")
  
  return hists

#prefit to data if needed
if options.toys>0 and options.toysFrequentist and not options.bypassFrequentistFit:  
  sess.run(nexpnomassign)
  minimize()
  xv = sess.run(x)

for itoy in range(ntoys):
  titoy[0] = itoy

  #reset all variables
  sess.run(globalinit)
  x.load(xv,sess)
    
  dofit = True
  
  if options.toys < 0:
    print("Running fit to asimov toy")
    sess.run(asimovassign)
    if options.randomizeStart:
      sess.run(asimovrandomizestart)
    else:
      dofit = False
  elif options.toys == 0:
    print("Running fit to observed data")
    sess.run(dataobsassign)
  else:
    print("Running toy %i" % itoy)  
    if options.toysFrequentist:
      #randomize nuisance constraint minima
      sess.run(frequentistassign)
    else:
      #randomize actual values
      sess.run(bayesassign)      
      
    outvalsgens,thetavalsgen = sess.run([outputs,theta])  
      
    if options.bootstrapData:
      #randomize from observed data
      sess.run(dataobsassign)
      sess.run(bootstrapassign)
    else:
      #randomize from expectation
      sess.run(toyassign)      

  #assign start values for nuisance parameters to constraint minima
  sess.run(thetastartassign)
  #set likelihood offset
  sess.run(nexpnomassign)
  
  if options.doBenchmark:
    neval = 10
    t0 = time.time()
    for i in range(neval):
      print(i)
      lval = sess.run([l])
    t = time.time() - t0
    print("%d l evals in %f seconds, %f seconds per eval" % (neval,t,t/neval))
        
    neval = 10
    t0 = time.time()
    for i in range(neval):
      print(i)
      lval,gval = sess.run([l,grad])
    t = time.time() - t0
    print("%d l+grad evals in %f seconds, %f seconds per eval" % (neval,t,t/neval))
        
    neval = 1
    t0 = time.time()
    for i in range(neval):
      hessval = sess.run([hessian])
    t = time.time() - t0
    print("%d hessian evals in %f seconds, %f seconds per eval" % (neval,t,t/max(1,neval)))
    
    exit()
  
  if options.saveHists and not options.toys > 1:
    nobsval = sess.run(nobs)
    obsHist = ROOT.TH1D('obs','',nbins,-0.5, float(nbins)-0.5)
    #array2hist(nobsval, obsHist)
    
    prefithists = fillHists('prefit')
  
  if dofit:
    minimize()

  #get fit output
  xval, outvalss, thetavals, theta0vals, nllval, nllvalfull = sess.run([x,outputs,theta,theta0,l,lfull])
  dnllval = 0.
  #get inverse hessians for error calculation (can fail if matrix is not invertible)
  try:
    invhessval,mineigval,isposdefval,edmval,invhessoutvals = sess.run([invhessian,mineigv,isposdef,edm,invhessianouts])
    errstatus = 0
  except:
    edmval = -99.
    isposdefval = False
    mineigval = -99.
    invhessoutvals = outvalss
    errstatus = 1
    
  if isposdefval and edmval > -edmtol:
    status = 0
  else:
    status = 1
  
  print("status = %i, errstatus = %i, nllval = %f, nllvalfull = %f, edmval = %e, mineigval = %e" % (status,errstatus,nllval,nllvalfull,edmval,mineigval))  
  
  if errstatus==0:
    fullsigmasv = np.sqrt(np.diag(invhessval))
    thetasigmasv = fullsigmasv[npoi:]
  else:
    thetasigmasv = -99.*np.ones_like(thetavals)
  
  thetaminosups = -99.*np.ones_like(thetavals)
  thetaminosdowns = -99.*np.ones_like(thetavals)
  
  outsigmass = []
  outminosupss = []
  outminosdownss = []
  outminosupd = {}
  outminosdownd = {}

  for output, outvals,invhessoutval in zip(outputs, outvalss,invhessoutvals):
    outname = ":".join(output.name.split(":")[:-1])    

    if not options.toys > 0:
      dName = 'asimov' if options.toys < 0 else 'data fit'
      correlationHist = ROOT.TH2D('correlation_matrix_channel'+outname, 'correlation matrix for '+dName+' in channel'+outname, int(nparms), 0., 1., int(nparms), 0., 1.)
      covarianceHist  = ROOT.TH2D('covariance_matrix_channel' +outname, 'covariance matrix for ' +dName+' in channel'+outname, int(nparms), 0., 1., int(nparms), 0., 1.)
      correlationHist.GetZaxis().SetRangeUser(-1., 1.)

      #set labels
      for ip1, p1 in enumerate(parms):
        correlationHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
        correlationHist.GetYaxis().SetBinLabel(ip1+1, '%s' % p1)
        covarianceHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
        covarianceHist.GetYaxis().SetBinLabel(ip1+1, '%s' % p1)

    if errstatus==0:
      parameterErrors = np.sqrt(np.diag(invhessoutval))
      sigmasv = parameterErrors[:npoi]
      if not options.toys > 0:
        variances2D     = parameterErrors[np.newaxis].T * parameterErrors
        correlationMatrix = np.divide(invhessoutval, variances2D)
        #array2hist(correlationMatrix, correlationHist)
        #array2hist(invhessoutval, covarianceHist)
    else:
      sigmasv = -99.*np.ones_like(outvals)
    
    minoserrsup = -99.*np.ones_like(sigmasv)
    minoserrsdown = -99.*np.ones_like(sigmasv)
    
    outsigmass.append(sigmasv)
    outminosupss.append(minoserrsup)
    outminosdownss.append(minoserrsdown)
  
    outminosupd[outname] = minoserrsup
    outminosdownd[outname] = minoserrsdown

  if options.saveHists and not options.toys > 1:
    postfithists = fillHists('postfit')

  for var in options.minos:
    print("running minos-like algorithm for %s" % var)
    if var in systs:
      erroutidx = systs.index(var)
      erridx = npoi + erroutidx
      minoserrsup = thetaminosups
      minoserrsdown = thetaminosdowns
      scanname = "x"
      outthetaval = xval
      sigmas = thetasigmasv
    else:
      outname = var.split("_")[-1]
      poi = "_".join(var.split("_")[:-1])
      if not outname in outidxs:
        raise Exception("Output not found")
      if not poi in pois:
        raise Exception("poi not found")
      
      outidx = outidxs[outname]
      
      scanname = outname
      erroutidx = pois.index(poi)
      erridx = erroutidx
      minoserrsup = outminosupss[outidx]
      minoserrsdown = outminosdownss[outidx]
      outthetaval = np.concatenate((outvalss[outidx],thetavals),axis=0)
      sigmasv = outsigmass[outidx]

      
    minosminimizer = minosminimizers[scanname]
    scanminimizer = scanminimizers[scanname]
    scanvar = scanvars[scanname]
    
    l0.load(nllval+0.5,sess)
    x0.load(outthetaval,sess)

    errdirv = np.zeros_like(outthetaval)
    errdirv[erridx] = 1.
    
    errdir.load(errdirv,sess)
    x.load(xval,sess)
    a.load(sigmasv[erroutidx],sess)
    scanminimizer.minimize(sess)
    minosminimizer.minimize(sess)
    xvalminosup, nllvalminosup = sess.run([scanvar,l])
    dxvalup = xvalminosup[erridx]-outthetaval[erridx]
    minoserrsup[erroutidx] = dxvalup

    errdir.load(-errdirv,sess)
    x.load(xval,sess)
    a.load(sigmasv[erroutidx],sess)
    scanminimizer.minimize(sess)
    minosminimizer.minimize(sess)
    xvalminosdown, nllvalminosdown = sess.run([scanvar,l])
    dxvaldown = -(xvalminosdown[erridx]-outthetaval[erridx])
    minoserrsdown[erroutidx] = dxvaldown
    
  tstatus[0] = status
  terrstatus[0] = errstatus
  tedmval[0] = edmval
  tnllval[0] = nllval
  tnllvalfull[0] = nllvalfull
  tdnllval[0] = dnllval
  tscanidx[0] = -1
  tndof[0] = x.shape[0]
  tndofpartial[0] = npoi
  
  for output,outvals,outsigmas,minosups,minosdowns,outgenvals,toutvals,touterrs,toutminosups,toutminosdowns,toutgenvals in zip(outputs,outvalss,outsigmass,outminosupss,outminosdownss,outvalsgens,toutvalss,touterrss,toutminosupss,toutminosdownss,toutgenvalss):
    outname = ":".join(output.name.split(":")[:-1])    
    for poi,outval,outma,minosup,minosdown,outgenval,toutval,touterr,toutminosup,toutminosdown,toutgenval in zip(pois,outvals,outsigmas,minosups,minosdowns,outgenvals,toutvals,touterrs,toutminosups,toutminosdowns,toutgenvals):
      toutval[0] = outval
      touterr[0] = outma
      toutminosup[0] = minosup
      toutminosdown[0] = minosdown
      toutgenval[0] = outgenval
      if itoy==0:
        print('%s_%s = %e +- %f (+%f -%f)' % (poi,outname,outval,outma,minosup,minosdown))

  for syst,thetaval,theta0val,sigma,minosup,minosdown,thetagenval, tthetaval,ttheta0val,tthetaerr,tthetaminosup,tthetaminosdown,tthetagenval in zip(systs,thetavals,theta0vals,thetasigmasv,thetaminosups,thetaminosdowns,thetavalsgen, tthetavals,ttheta0vals,tthetaerrs,tthetaminosups,tthetaminosdowns,tthetagenvals):
    tthetaval[0] = thetaval
    ttheta0val[0] = theta0val
    tthetaerr[0] = sigma
    tthetaminosup[0] = minosup
    tthetaminosdown[0] = minosdown
    tthetagenval[0] = thetagenval
    if itoy==0:
      print('%s = %f +- %f (+%f -%f) (%s_In = %f)' % (syst, thetaval, sigma, minosup,minosdown,syst,theta0val))
    
  tree.Fill()
  
  for var in options.scan:
    print("running profile likelihood scan for %s" % var)
    if var in systs:
      erroutidx = systs.index(var)
      erridx = npoi + erroutidx
      sigmasv = thetasigmasv
      scanname = "x"
      outthetaval = xval
    else:
      outname = var.split("_")[-1]
      poi = "_".join(var.split("_")[:-1])
      if not outname in outidxs:
        raise Exception("Output not found")
      if not poi in pois:
        raise Exception("poi not found")
      
      outidx = outidxs[outname]
      
      scanname = outname
      erroutidx = pois.index(poi)
      erridx = erroutidx
      sigmasv = outsigmass[outidx]
      outthetaval = np.concatenate((outvalss[outidx],thetavals),axis=0)
      
      
    scanminimizer = scanminimizers[scanname]
    
    x0.load(outthetaval,sess)
    
    errdirv = np.zeros_like(outthetaval)
    errdirv[erridx] = 1.
    errdir.load(errdirv,sess)
        
    dsigs = np.linspace(0.,options.scanRange,options.scanPoints)
    signs = [1.,-1.]
    
    for sign in signs:
      x.load(xval,sess)
      for absdsig in dsigs:
        dsig = sign*absdsig
        
        if absdsig==0. and sign==-1.:
          continue
        
        aval = dsig*sigmasv[erroutidx]
        
        a.load(aval,sess)
        scanminimizer.minimize(sess)
    
        scanoutvalss,scanthetavals, nllvalscan, nllvalscanfull = sess.run([outputs,theta,l,lfull])
        dnllvalscan = nllvalscan - nllval
                          
        tscanidx[0] = erridx
        tnllval[0] = nllvalscan
        tnllvalfull[0] = nllvalscanfull
        tdnllval[0] = dnllvalscan
        
        for outvals,toutvals in zip(scanoutvalss,toutvalss):
          for outval, toutval in zip(outvals,toutvals):
            toutval[0] = outval
        
        for thetaval, tthetaval in zip(scanthetavals,tthetavals):
          tthetaval[0] = thetaval

        tree.Fill()


f.Write()
f.Close()
