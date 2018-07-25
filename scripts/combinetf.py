#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import tensorflow as tf
import numpy as np
import h5py
import h5py_cache
from HiggsAnalysis.CombinedLimit.tfh5pyutils import maketensor
import scipy
import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from array import array

from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface,jacobian

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
hnorm = f['hnorm']
hlogkavg = f['hlogkavg']
hlogkhalfdiff = f['hlogkhalfdiff']

#infer some metadata from loaded information
dtype = hnorm.dtype
nbinsfull = hnorm.shape[-1]
nbinsmasked = len(maskedchans)
nbins = nbinsfull - nbinsmasked
nsyst = len(systs)
nproc = len(procs)
nsignals = len(signals)

#build tensorflow graph for likelihood calculation

#start by creating tensors which read in the hdf5 arrays (optimized for memory consumption)
#note that this does NOT trigger the actual reading from disk, since this only happens when the
#returned tensors are evaluated for the first time inside the graph
data_obs = maketensor(hdata_obs)
norm = maketensor(hnorm)
logkavg = maketensor(hlogkavg)
logkhalfdiff = maketensor(hlogkhalfdiff)

if options.nonNegativePOI:
  boundmode = 1
else:
  boundmode = 0

pois = []  
  
if options.POIMode == "mu":
  npoi = nsignals
  poidefault = options.POIDefault*np.ones([npoi],dtype=dtype)
  for signal in signals:
    pois.append(signal)
elif options.POIMode == "none":
  npoi = 0
  poidefault = np.empty([],dtype=dtype)
else:
  raise Exception("unsupported POIMode")

nparms = npoi + nsyst
parms = np.concatenate([pois,systs])

if boundmode==0:
  xpoidefault = poidefault
elif boundmode==1:
  xpoidefault = np.sqrt(poidefault)

print("nbins = %d, npoi = %d, nsyst = %d" % (data_obs.shape[0], npoi, nsyst))

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
elif boundmode == 1:
  poi = tf.square(xpoi)

#interpolation for asymmetric log-normal
twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)

#matrix encoding effect of nuisance parameters
#memory efficient version (do summation together with multiplication in a single tensor contraction step)
#this is equivalent to 
#alpha = tf.reshape(alpha,[-1,1,1])
#theta = tf.reshape(theta,[-1,1,1])
#logk = logkavg + alpha*logkhalfdiff
#logsnorm = theta*logk
alphatheta = alpha*theta
logsnorm = tf.einsum('i,ijk->jk',theta,logkavg) + tf.einsum('i,ijk->jk',alphatheta,logkhalfdiff)

snorm = tf.exp(logsnorm)

#vector encoding effect of signal strengths
if options.POIMode == "mu":
  r = poi
elif options.POIMode == "none":
  r = tf.ones([nsignals],dtype=dtype)

rnorm = tf.concat([r,tf.ones([nproc-nsignals],dtype=dtype)],axis=0)

#pnormfull = rnorm*snorm*norm
#nexpfull = tf.reduce_sum(pnormfull,axis=0)

#final expected yields per-bin including effect of signal
#strengths and nuisance parmeters
#memory efficient version (do summation together with multiplication in a single tensor contraction step)
#equivalent to (with some reshaping to explicitly match indices)
#rnorm = tf.reshape(rnorm,[-1,1])
#pnormfull = rnorm*snorm*norm
#nexpfull = tf.reduce_sum(pnormfull,axis=0)
nexpfull = tf.einsum('i,ij,ij->j',rnorm,snorm,norm)

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

#pnormmasked = pnormfull[:nsignals,nbins:]
#pmaskedexp = tf.reduce_sum(pnormmasked, axis=-1)
snormmasked = snorm[:nsignals,nbins:]
normmasked = norm[:nsignals,nbins:]
#pmaskedexp = tf.einsum('i,ij,ij->i',r,snormmasked,normmasked)
pmaskedexppartial = tf.einsum('ij,ij->i',snormmasked,normmasked)
pmaskedexp = r*pmaskedexppartial

#maskedexp = tf.reduce_sum(pnormmasked, axis=0,keepdims=True)
maskedexp = nexpfull[nbins:]

#if nbinsmasked>0:
  #pmaskedexpnorm = tf.reduce_sum(pnormmasked/maskedexp, axis=-1)
#else:
  #pmaskedexpnorm = pmaskedexp
#pmaskedexpnorm = tf.einsum('i,ij,ij,j->i',r,snormmasked,normmasked,tf.reciprocal(maskedexp))
pmaskedexpnormpartial = tf.einsum('ij,ij,j->i',snormmasked,normmasked,tf.reciprocal(maskedexp))
pmaskedexpnorm = r*pmaskedexpnormpartial
 
#name outputs
poi = tf.identity(poi, name=options.POIMode)
pmaskedexp = tf.identity(pmaskedexp, "pmaskedexp")
pmaskedexpnorm = tf.identity(pmaskedexpnorm, "pmaskedexpnorm")
 
outputs = []

outputs.append(poi)
if nbinsmasked>0:
  outputs.append(pmaskedexp)
  outputs.append(pmaskedexpnorm)

grad = tf.gradients(l,x,gate_gradients=True)[0]
hessian = jacobian(grad,x,gate_gradients=True,parallel_iterations=1,back_prop=False)
eigvals = tf.self_adjoint_eigvals(hessian)
mineigv = tf.reduce_min(eigvals)
isposdef = mineigv > 0.
invhessian = tf.matrix_inverse(hessian)
gradcol = tf.reshape(grad,[-1,1])
edm = 0.5*tf.matmul(tf.matmul(gradcol,invhessian,transpose_a=True),gradcol)

invhessianouts = []
for output in outputs:
  jacout = jacobian(tf.concat([output,theta],axis=0),x,gate_gradients=True,parallel_iterations=1,back_prop=False)
  invhessianout = tf.matmul(jacout,tf.matmul(invhessian,jacout,transpose_b=True))
  invhessianouts.append(invhessianout)

l0 = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
x0 = tf.Variable(np.zeros(x.shape,dtype=dtype),trainable=False)
a = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
errdir = tf.Variable(np.zeros(x.shape,dtype=dtype),trainable=False)
dlconstraint = l - l0

lb = np.concatenate((-np.inf*np.ones([npoi],dtype=dtype),-np.inf*np.ones([nsyst],dtype=dtype)),axis=0)
ub = np.concatenate((np.inf*np.ones([npoi],dtype=dtype),np.inf*np.ones([nsyst],dtype=dtype)),axis=0)

xtol = np.finfo(dtype).eps
edmtol = math.sqrt(xtol)
btol = 1e-8
minimizer = ScipyTROptimizerInterface(l, var_list = [x], var_to_bounds={x: (lb,ub)}, options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})

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
#asimovrandomizestart = tf.assign(x,tf.clip_by_value(tf.contrib.distributions.MultivariateNormalFullCovariance(x,invhess).sample(),lb,ub))
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

tedmval = array('f',[0.])
tree.Branch('edmval',tedmval,'edmval/F')

tnllval = array('f',[0.])
tree.Branch('nllval',tnllval,'nllval/F')

tdnllval = array('f',[0.])
tree.Branch('dnllval',tdnllval,'dnllval/F')

tchisq = array('f',[0.])
tree.Branch('chisq', tchisq, 'chisq/F')

tchisqpartial = array('f',[0.])
tree.Branch('chisqpartial', tchisqpartial, 'chisqpartial/F')

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
sess.run(globalinit)

xv = sess.run(x)

#set likelihood offset
sess.run(nexpnomassign)

outvalsgens,thetavalsgen = sess.run([outputs,theta])

#prefit to data if needed
if options.toys>0 and options.toysFrequentist and not options.bypassFrequentistFit:  
  sess.run(nexpnomassign)
  ret = minimizer.minimize(sess)
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
      raise Exception("Randomization of starting values is not currently implemented.")
      #sess.run(asimovrandomizestart)
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
  if dofit:
    ret = minimizer.minimize(sess)

  #get fit output
  xval, outvalss, thetavals, theta0vals, nllval = sess.run([x,outputs,theta,theta0,l])
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
  
  print("status = %i, errstatus = %i, nllval = %f, edmval = %e, mineigval = %e" % (status,errstatus,nllval,edmval,mineigval))  
  
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

    if errstatus==0:
      #jac = jaccomp.compute(sess)
      #jact = np.transpose(jac)
      #invhessoutval = np.matmul(jac,np.matmul(invhessval,jact))
      sigmasv = np.sqrt(np.diag(invhessoutval))[:npoi]
      if not options.toys > 0:
        parameterErrors = np.sqrt(np.diag(invhessoutval))
        variances2D     = parameterErrors[np.newaxis].T * parameterErrors
        correlationMatrix = np.divide(invhessoutval, variances2D)
        for ip1, p1 in enumerate(parms):
          for ip2, p2 in enumerate(parms):
            correlationHist.SetBinContent(ip1+1, ip2+1, correlationMatrix[ip1][ip2])
            correlationHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
            correlationHist.GetYaxis().SetBinLabel(ip2+1, '%s' % p2)
            covarianceHist.SetBinContent(ip1+1, ip2+1, invhessoutval[ip1][ip2])
            covarianceHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
            covarianceHist.GetYaxis().SetBinLabel(ip2+1, '%s' % p2)
    else:
      sigmasv = -99.*np.ones_like(outvals)
      if not options.toys > 0:
        for ip1, p1 in enumerate(parms):
          for ip2, p2 in enumerate(parms):
            correlationHist.SetBinContent(ip1+1, ip2+1, -1.)
            correlationHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
            correlationHist.GetYaxis().SetBinLabel(ip2+1, '%s' % p2)
            covarianceHist.SetBinContent(ip1+1, ip2+1, -1.)
            covarianceHist.GetXaxis().SetBinLabel(ip1+1, '%s' % p1)
            covarianceHist.GetYaxis().SetBinLabel(ip2+1, '%s' % p2)
    
    minoserrsup = -99.*np.ones_like(sigmasv)
    minoserrsdown = -99.*np.ones_like(sigmasv)
    
    outsigmass.append(sigmasv)
    outminosupss.append(minoserrsup)
    outminosdownss.append(minoserrsdown)
  
    outminosupd[outname] = minoserrsup
    outminosdownd[outname] = minoserrsdown

    if not options.toys > 0:
      correlationHist.Write()
      covarianceHist .Write()

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
    
        scanoutvalss,scanthetavals, nllvalscan = sess.run([outputs,theta,l])
        dnllvalscan = nllvalscan - nllval
                          
        tscanidx[0] = erridx
        tnllval[0] = nllvalscan
        tdnllval[0] = dnllvalscan
        
        for outvals,toutvals in zip(scanoutvalss,toutvalss):
          for outval, toutval in zip(outvals,toutvals):
            toutval[0] = outval
        
        for thetaval, tthetaval in zip(scanthetavals,tthetavals):
          tthetaval[0] = thetaval

        tree.Fill()


f.Write()
f.Close()