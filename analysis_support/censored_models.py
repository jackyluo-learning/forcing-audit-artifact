"""Interval-censored likelihoods; explicitly validated against lifelines.

Direct optimization exists to make 10,000 cluster refits practical. It changes
only the numerical solver, not the Weibull likelihood. No censoring is dropped.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr


def weibull_fit(x, lower, upper, weights=None, start=None):
    x=np.asarray(x,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
    logl=np.log(np.maximum(lower,1e-100)); finite=np.isfinite(upper)
    logu=np.log(np.where(finite,upper,1.0))
    def obj(p):
        a,b,t=p;shape=np.exp(t);mu=a+b*x
        zl=shape*(logl-mu);zu=shape*(logu-mu)
        if np.max(zl)>650 or np.max(zu[finite])>650:return 1e100,np.zeros(3)
        A=np.exp(np.clip(zl,-745,650));B=np.exp(np.clip(zu,-745,650))
        delta=np.maximum(B-A,1e-300)
        small=delta<50
        inv=np.zeros_like(delta);inv[small]=1/np.expm1(delta[small])
        ratio=np.zeros_like(delta);ratio[small]=delta[small]*inv[small]
        ll=-A;gmu=shape*A;gt=-A*zl
        ll[finite]+=np.log(-np.expm1(-delta[finite]))
        gmu[finite]-=shape*ratio[finite]
        gt[finite]+=(B[finite]*zu[finite]-A[finite]*zl[finite])*inv[finite]
        return -np.dot(w,ll),-np.array([np.dot(w,gmu),np.dot(w,gmu*x),np.dot(w,gt)])
    if start is None:
        start=np.array([np.average(np.log(np.where(finite,np.sqrt(np.maximum(lower,.5)*upper),lower)),weights=w),1.,1.])
    fit=minimize(obj,start,method='L-BFGS-B',jac=True,bounds=[(-10,10),(-40,40),(-4,4)],options={'ftol':1e-12,'gtol':1e-7,'maxiter':300,'maxls':100})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f'Weibull optimization failed: {fit.message}')
    return fit.x,float(fit.fun)


def normal_interval_fit(x,lower,upper,weights=None,start=None):
    """Secondary level-scale normal/Tobit likelihood; upper inf is right censoring."""
    x=np.asarray(x,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
    finite=np.isfinite(upper)
    def obj(p):
        a,b,t=p;sd=np.exp(t);mu=a+b*x
        zlo=(lower-mu)/sd;zhi=(upper-mu)/sd
        log_hi=log_ndtr(zhi);log_lo=log_ndtr(zlo)
        diff=np.minimum(log_lo-log_hi,-1e-15)
        logprob=log_hi+np.log(-np.expm1(diff))
        # Survival representation avoids cancellation in the positive tail.
        sel=(zlo>0)&finite
        sh=log_ndtr(-zlo[sel]);sl=log_ndtr(-zhi[sel]);logprob[sel]=sh+np.log(-np.expm1(np.minimum(sl-sh,-1e-15)))
        logprob[~finite]=log_ndtr(-zlo[~finite])
        return -float(np.dot(w,logprob))
    if start is None:start=np.array([np.average(np.where(finite,(lower+upper)/2,lower),weights=w),1.,2.])
    fit=minimize(obj,start,method='L-BFGS-B',bounds=[(-200,200),(-20,20),(-5,6)],options={'ftol':1e-11,'gtol':1e-6,'maxiter':300})
    if not fit.success or not np.isfinite(fit.fun):raise RuntimeError(f'Tobit optimization failed: {fit.message}')
    return fit.x,float(fit.fun)
