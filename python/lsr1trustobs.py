import numpy as np
import tensorflow as tf
import math

class SR1TrustExact:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    #self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.U = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.e = tf.Variable(tf.ones_like(var),trainable=False)
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None:
      e,U = tf.self_adjoint_eig(B)
      #alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.e,e))
      alist.append(tf.assign(self.U,U))
    return tf.group(alist)
  
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    #eta = 0.
    eta = 0.15
    
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
        
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    def hesspexact(v):
      return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    def hesspapprox(B,v):
      return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    def doSR1Scaling(Bin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      return (B,H,tf.constant(False))
    
    def doSR1Update(ein,Uin,yin,dxin):
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      ecol = tf.reshape(ein,[-1,1])
      #Bx = tf.matmul(Bin,dx)
      #dyBx = y - Bx
      #den = tf.matmul(dyBx,dx,transpose_a=True)
            
      #dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      #dentest = tf.less(tf.abs(den),1e-8*dennorm)
      #dentest = tf.reshape(dentest,[])
      #dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      
      UTdx = tf.matmul(Uin, dx,transpose_a=True)
      UTy = tf.matmul(Uin,y,transpose_a=True)
      denalt = tf.matmul(y,dx,transpose_a=True) - tf.matmul(UTdx,ecol*UTdx,transpose_a=True)
      den = denalt
      dyBxalt =  UTy - ecol*UTdx
      dyBxaltnormsq = tf.reduce_sum(tf.square(dyBxalt))
      dyBxaltnorm = tf.sqrt(dyBxaltnormsq)
      dxnorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))
      dennorm = dxnorm*dyBxaltnorm
      dentest = tf.less(tf.abs(denalt),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      
      
      def doUpdate():
        #deltaB = tf.matmul(dyBx,dyBx,transpose_b=True)/den
        #dyBxmagsq = tf.reduce_sum(tf.square(dyBx))
        ##xisq = tf.square(dyBx)/dyBxmagsq
        ##z = tf.sqrt(xisq)
        #b = dyBx/tf.sqrt(dyBxmagsq)
        ##z = tf.matmul(Uin,tf.reshape(b,[-1,1]),transpose_a=True)
        ##z = tf.matrix_solve(Uin,b)
        #z = tf.matmul(Uin, b, transpose_a=True)
              
        zalt = dyBxalt/dyBxaltnorm
        z = zalt
        
        #z = tf.Print(z,[z],message="z")
        #z = tf.Print(z,[zalt],message="zalt")
        #z = tf.Print(z,[den],message="den")
        #z = tf.Print(z,[denalt],message="denalt")
      
        #rho = dyBxmagsq/den
        #rho = tf.reshape(rho,[])
        #signedrho = rho
        #rho = tf.abs(rho)
        
        signedrho = dyBxaltnormsq/denalt
        signedrho = tf.reshape(signedrho,[])
        rho = tf.abs(signedrho)
      
        C = tf.diag(ein) + rho*tf.matmul(z,z,transpose_b=True)
        normC = tf.sqrt(tf.reduce_sum(tf.square(C)))
      
        xisq = tf.square(z)
        #zmag = tf.sqrt(tf.reduce_sum(xisq))
        #xisq = tf.Print(xisq,[zmag],message = "zmag")
        #rho = tf.sqrt(dyBxmagsq/tf.abs(den))
        #rho = dyBxmagsq/tf.abs(den)

        
        #deltaBalt = rho*tf.matmul(z,z,transpose_b=True)
        #deltaB = tf.Print(deltaB,[deltaB],message = "deltaB")
        #deltaB = tf.Print(deltaB,[deltaBalt],message = "deltaBalt")
        
        #rho = tf.Print(rho,[z],message = "z",summarize=1000)
        #rho = tf.Print(rho,[rho],message = "rho")
        #rho = tf.Print(rho,[den],message = "den")
        
        
        flipsign = tf.reshape(den,[])<0.
        #rho = tf.where(flipsign,-rho, rho)
        #d = tf.where(flipsign, -ein, ein)
        d = ein
        dalt = -tf.reverse(d, axis=(0,))
        #dalt = tf.concat([dalt[1:],-tf.reshape(dalt[0],[-1])],axis=0)
        d = tf.where(flipsign, dalt, d)
        #d = tf.Print(d,[d],message="d",summarize=1000)
        
        xisqalt = tf.reverse(xisq,axis=(0,))
        xisq = tf.where(flipsign,xisqalt,xisq)
        
        #Hy = tf.matmul(Hin,y)
        #dxHy = dx - Hy
        #deltaH = tf.matmul(dxHy,dxHy,transpose_b=True)/tf.matmul(dxHy,y,transpose_a=True)
        
        #Bd = tf.matrix_band_part(Bin,0,0)
        #Bd = Bin
        
        #etrueold = tf.self_adjoint_eigvals(Bin)
        #etrueold = tf.diag_part(Bd)
        
        #B = Bin + deltaB
        #B = Bin
        

        
        #H = Hin + deltaH
        
        #etruenew = tf.self_adjoint_eigvals(B)
        
        #wtrue = 1. + signedrho*tf.reduce_sum(tf.reshape(xisq,[1,-1])/(tf.reshape(etrueold,[1,-1]) - tf.reshape(etruenew,[-1,1])),axis=-1)
        
        #B = tf.Print(B,[wtrue],message = "wtrue",summarize=1000)

        
        def eUpdate():
          
          
          #t0 = tf.zeros_like(var)+1e-6
          #t0 = tf.reshape(xisq,[-1])
          w0 = tf.ones_like(var)
          thres0 = tf.zeros_like(var)
          etaw = 1e-14
          
          en1 = d[:-1]
          e1 = d[1:]
          delta = (e1 - en1)/rho
          xisqn1 = tf.reshape(xisq[:-1],[-1])
          xisq1 = tf.reshape(xisq[1:],[-1])
          
          di = tf.reshape(d,[-1,1])
          dj = tf.reshape(d,[1,-1])
          deltam = (dj-di)/rho
          deltamn1 = deltam[:-1]
          
          #t0n1 = 1e-2*delta
          
          s0mden = deltamn1 - tf.reshape(delta,[-1,1])
          #s0m = tf.reshape(xisqn1,[-1,1])/s0mden
          s0m = tf.reshape(xisq,[1,-1])/s0mden
          s0m = tf.where(tf.equal(s0mden,0.),tf.zeros_like(s0m),s0m)
          s0m = s0m - tf.matrix_band_part(s0m,1,0)
          s0 = tf.reduce_sum(s0m,axis=-1)
          
          a0 = 1.+s0
          b0 = -(xisqn1 + xisq1 + (1.+s0)*delta)
          c0 = xisqn1*delta
          t0n1 = (-b0 - tf.sqrt(tf.square(b0) - 4.*a0*c0))/(2*a0)
          #t0n1 = tf.where(tf.is_nan(t0n1),0.5*delta,t0n1)
          t0n1 = tf.where(tf.is_nan(t0n1),tf.zeros_like(t0n1),t0n1)
          #t0n1 = tf.where(tf.is_inf(t0n1),0.5*delta,t0n1)
          
          #t0n = 0.1*tf.ones([1],dtype=var.dtype)
          #t0n = tf.zeros([1],dtype=var.dtype)
          t0n = tf.reshape(xisq[-1],[-1])
          t0 = tf.concat([t0n1,t0n],axis=0)          
          
          #sq0 = tf.square(b0) - 4*a0*c0
          
          #t0 = tf.Print(t0,[s0],message = "s0",summarize=1000)
          #t0 = tf.Print(t0,[a0],message = "a0",summarize=1000)
          #t0 = tf.Print(t0,[b0],message = "b0",summarize=1000)
          #t0 = tf.Print(t0,[c0],message = "c0",summarize=1000)
          #t0 = tf.Print(t0,[sq0],message = "sq0",summarize=1000)
          #t0 = tf.Print(t0,[t0],message = "t0",summarize=1000)
          
          loop_vars = [t0,tf.constant(True),tf.constant(0)]
          def cond(t,unconverged,j):
            return (unconverged) & (j<50)
            #return (tf.sqrt(tf.reduce_sum(tf.square(w))) > 1e-10) & (j<100)
          
          
          def body(t,unconverged,j):
            psim = tf.reshape(xisq,[1,-1])/(deltam - tf.reshape(t,[-1,1]))
            #psim = xisq/(deltam + t)
            psi = tf.diag_part(tf.cumsum(psim,axis=-1))
            psifull = tf.reduce_sum(psim,axis=-1)
            phi = psifull - psi
            #phi = tf.where(tf.is_nan(phi),tf.zeros_like(phi),phi)
            w = 1. + phi + psi
            w = tf.where(tf.is_nan(w),tf.zeros_like(w),w)
            #wfull = 1. + rho*tf.reduce_sum(tf.reshape(xisq,[1,-1])/(tf.reshape(d,[1,-1]) - tf.reshape(d+rho*t,[-1,1])),axis=-1)
            
            #w = tf.Print(w, [w], summarize=5, message = "w")
            #w = tf.Print(w, [wfull], summarize=1000, message = "wfull")
            
            #w = tf.Print(w, [psi], summarize=5, message = "psi")
            #w = tf.Print(w, [psifull], summarize=5, message = "psifull")
            #w = tf.Print(w, [phi], summarize=5, message = "phi")
            
            psiprimem = tf.reshape(xisq,[1,-1])/tf.square(deltam - tf.reshape(t,[-1,1]))
            psiprime = tf.diag_part(tf.cumsum(psiprimem, axis=-1))
            psiprimefull = tf.reduce_sum(psiprimem,axis=-1)
            phiprime = psiprimefull - psiprime
            #phiprime = tf.where(tf.is_nan(phiprime),tf.zeros_like(phiprime),phiprime)
            
            #first n-1 terms
            psin1 = psi[:-1]
            phin1 = phi[:-1]
            psiprimen1 = psiprime[:-1]
            phiprimen1 = phiprime[:-1]
            wn1 = w[:-1]
            tn1 = t[:-1]
            Delta = delta - tn1

            c = 1. + phin1 - Delta*phiprimen1
            b = (Delta*wn1*psin1)/(psiprimen1*c)
            a = (Delta*(1.+phin1)+tf.square(psin1)/psiprimen1)/c + psin1/psiprimen1
            tn1 = tn1 + 2.*b/(a + tf.sqrt(tf.square(a) - 4.*b))
            #tn1 = tf.where(tf.is_nan(tn1),0.5*delta,tn1)
            #tn1 = tf.where(tf.is_nan(tn1),tf.zeros_like(tn1),tn1)
            #tn1 = tf.where(tf.is_inf(tn1),0.5*delta,tn1)
            
            #tn1 = tf.Print(tn1,[delta],message = "delta",summarize=1000)
            #tn1 = tf.Print(tn1,[t],message = "t",summarize=1000)
            #tn1 = tf.Print(tn1,[Delta],message = "Delta",summarize=1000)
            #tn1 = tf.Print(tn1,[w],message = "w",summarize=1000)
            #tn1 = tf.Print(tn1,[psi],message = "psi",summarize=1000)
            #tn1 = tf.Print(tn1,[psiprime],message = "psiprime",summarize=1000)
            #tn1 = tf.Print(tn1,[c],message = "c",summarize=1000)
            #tn1 = tf.Print(tn1,[b],message = "b",summarize=1000)
            #tn1 = tf.Print(tn1,[a],message = "a",summarize=1000)
            
            #last term
            psin = psi[-1]
            psiprimen = psiprime[-1]
            tn = t[-1] + (1.+psin)*psin/psiprimen
            tn = tf.reshape(tn,[-1])
            
            t = tf.concat([tn1,tn],axis=0)
            t = tf.where(tf.is_nan(t),tf.zeros_like(t),t)
            
            thres = normC*tf.sqrt(phiprime + psiprime)/rho
            thres = tf.where(tf.is_nan(thres),tf.ones_like(thres),thres)
            unconverged = tf.reduce_any(tf.abs(w) > etaw*thres)
            
            
            #t = tf.Print(t,[t],message = "t", summarize=1000)
            #t = tf.Print(t,[w],message="w", summarize=1000)
            magw = tf.sqrt(tf.reduce_sum(tf.square(w)))

            
            t = tf.Print(t,[magw],message="magw")
            
            maxconv = tf.reduce_max(tf.abs(w)/thres)
            t = tf.Print(t,[maxconv],message="maxconv")
            
            return (t,unconverged,j+1)
            
            
          t,unconverged,j = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
          #t = tf.Print(t,[t[0],w],message = "t0,w")
          #t = tf.Print(t,[w],message = "w",summarize=5)
          #t = tf.Print(t,[t],message = "t",summarize=5)
          eout = d + rho*t
          eoutalt = -tf.reverse(eout,axis=(0,))
          #eoutalt = d - rho*t
          #eoutalt = -(tf.reverse(d,axis=(0,)) + rho*t)
          #eoutalt = -tf.reverse(d,axis=(0,)) - rho*t
          eout = tf.where(flipsign,eoutalt,eout)
          
          talt = tf.reverse(t,axis=(0,))
          t = tf.where(flipsign,talt,t)
          
          #now compute eigenvectors
          #Dinv = tf.reciprocal(tf.reshape(ein,[1,-1]) - tf.reshape(eout,[-1,1]))
          #Dinvz = Dinv*tf.reshape(z,[1,-1])
          #Dinvzmag = tf.sqrt(tf.reduce_sum(tf.square(Dinvz),axis=-1))
          #uout = tf.matmul(Uin,Dinvz,transpose_b=True)/Dinvzmag
          
          #Dinv = tf.reciprocal(tf.reshape(ein,[1,-1]) - tf.reshape(eout,[-1,1]))
          #Dinvz = Dinv*tf.reshape(z,[1,-1])
          #Dinvzmag = tf.sqrt(tf.reduce_sum(tf.square(Dinvz),axis=-1))
          ##Dinvzmag = tf.where(tf.logical_or(tf.is_nan(Dinvzmag),tf.is_inf(Dinvzmag)), tf.ones_like(Dinvzmag), Dinvzmag)
          ##Dinvz = tf.where(tf.logical_or(tf.is_nan(Dinvz),tf.is_inf(Dinvz)), tf.ones_like(Dinvz), Dinvz)
          ##Dinvzscaled = Dinvz/Dinvzmag
          ##Dinvzscaled = tf.where(tf.is_nan(Dinvzscaled),tf.ones_like(Dinvzscaled),Dinvzscaled)
          ##uout = tf.matmul(Uin,Dinvzscaled,transpose_b=True)
          #uout = tf.matmul(Uin,Dinvz,transpose_b=True)/Dinvzmag
          
          ei = tf.reshape(ein,[-1,1])
          ej = tf.reshape(ein,[1,-1])
          D = (ej-ei)/signedrho - tf.reshape(t,[-1,1])
          
          ##Dinv = tf.reciprocal(tf.reshape(ein,[1,-1]) - tf.reshape(eout,[-1,1]))
          ##D = deltam - tf.reshape(t,[-1,1])
          #D = tf.reshape(ein,[1,-1]) - tf.reshape(eout,[-1,1])
          Dinv = tf.reciprocal(D)
          Dinvz = Dinv*tf.reshape(z,[1,-1])
          Dinvzmag = tf.sqrt(tf.reduce_sum(tf.square(Dinvz),axis=-1))
          ##Dinvzmag = tf.where(tf.logical_or(tf.is_nan(Dinvzmag),tf.is_inf(Dinvzmag)), tf.ones_like(Dinvzmag), Dinvzmag)
          ##Dinvz = tf.where(tf.logical_or(tf.is_nan(Dinvz),tf.is_inf(Dinvz)), tf.ones_like(Dinvz), Dinvz)
          ##Dinvzscaled = Dinvz/Dinvzmag
          ##Dinvzscaled = tf.where(tf.is_nan(Dinvzscaled),tf.ones_like(Dinvzscaled),Dinvzscaled)
          ##uout = tf.matmul(Uin,Dinvzscaled,transpose_b=True)
          uout = tf.matmul(Uin,Dinvz,transpose_b=True)/Dinvzmag
          uout = tf.where(tf.is_nan(uout),Uin,uout)
          
          #umask = tf.reshape(tf.equal(t,0.),[-1,1])
          
          #uisnancol = tf.reduce_any(tf.logical_or(tf.is_nan(uout),tf.is_inf(uout)),axis=0,keepdims=True)
          #ufalse = tf.constant(False,shape=uout.shape)
          #umask = tf.logical_or(umask,ufalse)
          #uout = tf.where(umask,Uin,uout)
          #uisnancol = tf.logical_or(uisnancol,ufalse)
          #uout = tf.where(uisnancol,Uin,uout)
          
          #uout = tf.where(tf.is_nan(uout),Uin,uout)
          

          #eout = tf.where(flipsign,-eout,eout)
          return (eout,uout)
        
        e,U = eUpdate()
        #U = Uin
        
        return (e,U)
      
      e,U = tf.cond(dentest, lambda: (ein,Uin), doUpdate)
      
      return (e,U)
    
    #grad = self.grad
    #B = self.B
    esec = self.e
    Usec = self.U
    
    #esec = tf.Print(esec,[esec],summarize=1000,message="esecin")
        
    #dgrad = grad - self.grad_old
    #dx = var - self.var_old
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    esec,Usec = tf.cond(self.doiter_old, lambda: doSR1Update(esec,Usec,dgrad,dx), lambda: (esec,Usec))  
    
    #nprint = 5
    #etrue,Utrue = tf.self_adjoint_eig(B)
    #esec = tf.Print(esec,[esec],message = "esec",summarize=nprint)
    #esec = tf.Print(esec,[etrue],message = "etrue",summarize=nprint)
    
    #esec = tf.Print(esec,[Usec[:,0]],message = "Usec",summarize=10000)
    #esec = tf.Print(esec,[Utrue[:,0]],message = "Utrue",summarize=10000)
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():

      lam = esec
      U = Usec

      #lam,U = tf.self_adjoint_eig(B) #TODO: check if what is returned here should actually be UT in the paper
      #U = tf.transpose(U)

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      a = tf.matmul(U, gradcol,transpose_a=True)
      a = tf.reshape(a,[-1])
      #a = tf.Print(a,[U],message="U",summarize=10000)
      #a = tf.Print(a,[gradcol],message="gradcol",summarize=10000)
      #a = tf.Print(a,[a],message="a",summarize=10000)
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      a = tf.Print(a,[amagsq,gmagsq],message = "amagsq,gmagsq")
      
      #a = tf.matmul(U, gradcol,transpose_a=False)
      asq = tf.square(a)
      

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      #abarsq = tf.Print(abarsq, [a, abar, lam, lambar], message = "a,abar,lam,lambar")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[usesolu], message="usesolu")
      #usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      #def solu():
        #return -tf.matmul(H,gradcol)
      
      def sigma():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        #sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          #phi = tf.Print(phi,[phi],message = "checking phi in cond()")
          return tf.logical_and(phi < -tol, j<maxiter)
          #return tf.logical_and(tf.abs(phi) > tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          #sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[sigmaiter,phiiter],message = "sigmaiter,phiiter")
        
        #coeffs = -a/(lam+sigmaiter)
        #coeffs = tf.reshape(coeffs,[1,-1])
        #p = tf.reduce_sum(coeffs*U, axis=-1)
        
        return sigmaiter
      
      #p = tf.cond(usesolu, solu, sol)
      sigma = tf.cond(usesolu, lambda: tf.zeros([],dtype=var.dtype), sigma)
      #p = tf.reshape(p,[-1])
      
      coeffs = -a/(lam+sigma)
      coeffs = tf.reshape(coeffs,[1,-1])
      p = tf.reduce_sum(coeffs*U, axis=-1)
      #magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      #p = tf.Print(p,[magp],message = "magp")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(a*coeffs) + 0.5*tf.reduce_sum(lam*tf.square(coeffs)))

      #predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(B,tf.reshape(p,[-1,1])),[-1])*p) )
      
      #predicted_reduction_out = predicted_reduction_out_alt
      
      #predicted_reduction_out = tf.Print(predicted_reduction_out,[predicted_reduction_out],message = "predicted_reduction_out")
      #predicted_reduction_out = tf.Print(predicted_reduction_out,[predicted_reduction_out_alt],message = "predicted_reduction_out_alt")
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    doiter = tf.Print(doiter,[doiter],message="doiter")
    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      #alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.e,esec)) 
      alist.append(tf.assign(self.U,Usec)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]





class SR1TrustOBS:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None, H = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None and H is not None:
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
    return tf.group(alist)
  
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    #eta = 0.
    eta = 0.15
    
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    def hesspexact(v):
      return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    def hesspapprox(B,v):
      return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    def doSR1Scaling(Bin,Hin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      H = Hin/scale
      return (B,H,tf.constant(False))
    
    def doSR1Update(Bin,Hin,yin,dxin):
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      Bx = tf.matmul(Bin,dx)
      dyBx = y - Bx
      den = tf.matmul(dyBx,dx,transpose_a=True)
      deltaB = tf.matmul(dyBx,dyBx,transpose_b=True)/den
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.less(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      deltaB = tf.where(dentest,tf.zeros_like(deltaB),deltaB)
      #deltaB = tf.where(self.doiter_old, deltaB, tf.zeros_like(deltaB))
      
      Hy = tf.matmul(Hin,y)
      dxHy = dx - Hy
      deltaH = tf.matmul(dxHy,dxHy,transpose_b=True)/tf.matmul(dxHy,y,transpose_a=True)
      deltaH = tf.where(dentest,tf.zeros_like(deltaH),deltaH)
      #deltaH = tf.where(self.doiter_old, deltaH, tf.zeros_like(deltaH))
      
      B = Bin + deltaB
      H = Hin + deltaH
      return (B,H)
    
    #grad = self.grad
    B = self.B
    H = self.H
    
    #dgrad = grad - self.grad_old
    #dx = var - self.var_old
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    B,H = tf.cond(self.doiter_old, lambda: doSR1Update(B,H,dgrad,dx), lambda: (B,H))  
    
  
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():

      lam,U = tf.self_adjoint_eig(B) #TODO: check if what is returned here should actually be UT in the paper
      #U = tf.transpose(U)

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      a = tf.matmul(U, gradcol,transpose_a=True)
      a = tf.reshape(a,[-1])
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      a = tf.Print(a,[amagsq,gmagsq],message = "amagsq,gmagsq")
      
      #a = tf.matmul(U, gradcol,transpose_a=False)
      asq = tf.square(a)
      

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      abarsq = tf.Print(abarsq, [a, abar, lam, lambar], message = "a,abar,lam,lambar")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      def solu():
        return -tf.matmul(H,gradcol)
      
      def sol():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          #phi = tf.Print(phi,[phi],message = "checking phi in cond()")
          return tf.logical_and(phi < -tol, j<maxiter)
          #return tf.logical_and(tf.abs(phi) > tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[sigmaiter,phiiter],message = "sigmaiter,phiiter")
        
        coeffs = -a/(lam+sigmaiter)
        coeffs = tf.reshape(coeffs,[1,-1])
        p = tf.reduce_sum(coeffs*U, axis=-1)
        
        return p
      
      p = tf.cond(usesolu, solu, sol)
      p = tf.reshape(p,[-1])
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      p = tf.Print(p,[magp],message = "magp")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(B,tf.reshape(p,[-1,1])),[-1])*p) )
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]




class LSR1TrustOBS:
    
  def __init__(self, loss, var,grad, k=100, initialtrustradius = 1.):
    self.k = k
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    #self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.MpsiT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype),trainable = False, validate_shape=False)
    #self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([],dtype=var.dtype)
    self.gamma = tf.Variable(tf.ones([],dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(False)
    self.updateidx = tf.Variable(tf.zeros([],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
  def initialize(self, loss, var, grad):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    return tf.group(alist)
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    eta = 0.
    #eta = 0.15
    #tau1 = 0.1
    #tau2 = 0.3

    #defaults from nocedal and wright
    #eta = 1e-3
    tau1 = 0.1
    tau2 = 0.75
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    #trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    trustradius_out = tf.where(tf.less(rho,tau1),0.5*self.trustradius,
                               tf.where(tf.less(rho,tau2), self.trustradius,
                               tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    
    
    def doSR1Scaling(yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      return (scale,False)
    
    gamma,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(dgrad,dx), lambda: (self.gamma,self.doscaling))

    
    def doSR1Update(STin,YTin,yin,dxin):
      ycol = tf.reshape(yin,[-1,1])
      dxcol = tf.reshape(dxin,[-1,1])
      
      yrow = tf.reshape(yin,[1,-1])
      dxrow = tf.reshape(dxin,[1,-1])
      
      #dyBx = ycol - Bv(gamma,self.psi,self.M,dxcol)
      dyBx = ycol - Bv(gamma,self.psi,self.MpsiT,dxcol)
      den = tf.matmul(dyBx, dxcol, transpose_a = True)
      #den = tf.reshape(den,[])
      
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.greater(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      nonzero = dentest
      #nonzero = tf.logical_and(dentest,tf.not_equal(actual_reduction,0.))
      
      #nonzero = tf.Print(nonzero, [den,dennorm, dentest, nonzero], message = "den, dennorm, dentest, nonzero")
      
      #nonzero = tf.abs(den) > 1e-8
      
      #doappend = tf.logical_and(nonzero, tf.shape(STin)[0] < self.k)
      #doreplace = tf.logical_and(nonzero, tf.shape(STin)[0] >= self.k)
      
      sliceidx = tf.where(tf.shape(STin)[0] < self.k, 0, 1)
      
      #print(den.shape)
      
      def update():
        ST = tf.concat([STin[sliceidx:],dxrow],axis=0)
        YT = tf.concat([YTin[sliceidx:],yrow],axis=0)
        return (ST,YT)
      
      ST,YT = tf.cond(nonzero, update, lambda: (STin, YTin))

      return (ST,YT)
    
    ST = self.ST
    YT = self.YT
        

    #doscaling = tf.constant(False)
    ST,YT = tf.cond(self.doiter_old, lambda: doSR1Update(ST,YT,dgrad,dx), lambda: (ST,YT))    
    
    #compute compact representation
    S = tf.transpose(ST)
    Y = tf.transpose(YT)
    psi = Y - gamma*S
    psiT = tf.transpose(psi)
    STY = tf.matmul(ST,YT,transpose_b=True)
    D = tf.matrix_band_part(STY,0,0)
    L = tf.matrix_band_part(STY,-1,0) - D
    LT = tf.transpose(L)
    STB0S = gamma*tf.matmul(ST,S)
    Minverse = D + L + LT - STB0S
    MpsiT = tf.matrix_solve(Minverse,psiT)
    #M = tf.matrix_inverse(Minverse)
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():
      #grad = self.grad
      
      #compute eigen decomposition
      #psiTpsi = tf.matmul(psiT,psi)
      #epsiTpsi = tf.self_adjoint_eigvals(psiTpsi)
      #e0psiTpsi = tf.reduce_min(epsiTpsi)
      #psiTpsi = tf.Print(psiTpsi,[e0psiTpsi], message = "e0psiTpsi")
      ##psiTpsi = psiTpsi + 4.*tf.maximum(-e0psiTpsi,tf.zeros_like(e0psiTpsi))*tf.eye(tf.shape(psiTpsi)[0],dtype=psiTpsi.dtype)
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      #def chol():
        #RT = tf.cholesky(psiTpsi)
        #R = tf.transpose(RT)
        #return (R,RT)
      
      #def qr():
        #Q,R = tf.qr(psi)
        #RT = tf.transpose(R)
        #return (R,RT)
      
      ##R,RT = tf.cond(e0psiTpsi > 0., chol, qr)
      #R,RT = chol()
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      
      Q,R = tf.qr(psi)
      detR = tf.matrix_determinant(R)
      #assertR = tf.Assert(tf.not_equal(detR,0.),[detR])
      #with tf.control_dependencies([assertR]):
        #R = tf.Print(R,[detR],message="detR")
      
      RT = tf.transpose(R)
      MRT = tf.matrix_solve(Minverse,RT)
      RMRT = tf.matmul(R,MRT)
      #RMRT = tf.matmul(R,tf.matmul(M,R,transpose_b=True))

      e,U = tf.self_adjoint_eig(RMRT) #TODO: check if what is returned here should actually be UT in the paper
      

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      #gpll = tf.matmul(tf.matmul(psi,tf.matmul(Rinverse,U)), gradcol,transpose_a=True)
      #gpll = tf.matmul(tf.matmul(psi,tf.matrix_solve(R,U)), gradcol,transpose_a=True)
      gpll = tf.matmul(tf.matmul(Q,U), gradcol,transpose_a=True)
      gpll = tf.reshape(gpll,[-1])
      gpllsq = tf.square(gpll)
      gmagsq = tf.reduce_sum(tf.square(grad))
      gpllmagsq = tf.reduce_sum(gpllsq)
      gperpmagsq = gmagsq - gpllmagsq
      gperpmagsq = tf.maximum(gperpmagsq,tf.zeros_like(gperpmagsq))
      gperpmagsq = tf.reshape(gperpmagsq,[1])
      
      #gpll = tf.Print(gpll,[tf.shape(gpll)], message = "gpll shape:")
      a = gpll
      a = tf.concat([a,tf.sqrt(gperpmagsq)],axis=0)
      #a = a[:var.shape[0]]
      asq = tf.square(a)
      
      #lam = e + gamma
      #lam = tf.concat([lam,tf.reshape(gamma,[1])],axis=0)
      lam = tf.pad(e,[[0,1]]) + gamma
      #lam = lam[:var.shape[0]]

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = tf.minimum(lam[0],gamma)
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      abarsq = tf.Print(abarsq, [a, abar, lam, lambar,gperpmagsq,gmagsq,gpllmagsq], message = "a,abar,lam,lambar,gperpmagsq,gmagsq, gpllmagsq")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      
      def sigma_sol():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          return tf.logical_and(phi < -tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        
        return sigmaiter
      
      sigma = tf.cond(usesolu, lambda: sigma0, lambda: sigma_sol())
      tau = sigma + gamma
      
      #print(var.shape[0])
      I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      innerinverse = tau*Minverse + tf.matmul(psi,psi,transpose_a=True)
      innerpsiT = tf.matrix_solve(innerinverse,psiT)
      #inner = tf.matrix_inverse(innerinverse)
      #inner2 = tf.matmul(tf.matmul(psi,inner),psi, transpose_b=True)
      inner2 = tf.matmul(psi,innerpsiT)
      p = -tf.matmul(I-inner2, gradcol)/tau
      p = tf.reshape(p,[-1])
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      detMinverse = tf.matrix_determinant(Minverse)
      detinnerinverse = tf.matrix_determinant(innerinverse)
      p = tf.Print(p,[magp,detR,detMinverse,detinnerinverse],message = "magp, detR, detMinverse, detinnerinverse")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(Bvflat(gamma, psi, MpsiT, p)*p))
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.ST,ST,validate_shape=False))
      alist.append(tf.assign(self.YT,YT,validate_shape=False))
      alist.append(tf.assign(self.psi,psi,validate_shape=False))
      #alist.append(tf.assign(self.M,M,validate_shape=False))
      alist.append(tf.assign(self.MpsiT,MpsiT,validate_shape=False))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.gamma,gamma)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]

