# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:09:23 2020

@author: HEKIMOGL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:06:54 2020

@author: HEKIMOGL
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:46:19 2020

@author: Administrator
"""
import numpy as np
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
import scipy as sc
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import fmin_bfgs,fmin_powell,fmin_slsqp,minimize
from scipy.integrate import quad,trapz,quadrature,fixed_quad,simpson
from yahoo_finance import Share
import yfinance as yf
from pandas_datareader import data as pdr
from joblib import Parallel, delayed
from multiprocessing import Pool
#import pandas_datareader as pdrd
import quandl as qdl
import datetime as dt
import numexpr as ne
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import math
#stocklist=['DB','ENI.MI','^XU100']
#os.chdir('C:\\Users\\Administrator\\Desktop\\abdl\\calibre_HCJ')
def datagetyahoo(stock,start,end):
    ydr = pdrd.get_data_yahoo(stock,start,end)
    #YahooDailyReader(stock,start,end)
    df= ydr['Adj Close']
    return df

def NIGmom(x):
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    k=K/3.0-1
    A=S/(3.0*K)
    sgm=np.sqrt(V/(1+A**2*k))
    theta=sgm*A
    mu=E-theta
    return  abs(sgm),abs(k),theta,mu
def NIGddnew(x,sgm,k,theta,mu):
    kp=1./k
    A=theta/sgm**2
    B=np.sqrt(theta**2+sgm**2*kp)/sgm**2
    C=1.0/np.pi*np.exp(kp)*np.sqrt(theta**2*kp/sgm**2+kp**2)
    f=C*(np.exp(A*(x-mu))*sc.special.kv(1,B*np.sqrt((x-mu)**2+kp*sgm**2))/np.sqrt((x-mu)**2+kp*sgm**2))
    #(np.sqrt(2)/(np.sqrt(np.pi)*sgm*sc.special.gamma(nu)))*(np.abs(x-mus)/np.sqrt(theta**2+2*sgm**2))**(nu-0.5)*np.exp((x-mus)*theta/sgm**2)*sc.special.kv(nu-0.5,np.abs(x-mus)*np.sqrt(theta**2+2*sgm**2)/sgm**2)
    return f
def brloglikNIG(pars,x):
    sgm=pars[0];k=pars[1];theta=pars[2];mu=pars[3];
    Lv=np.sum(np.log(NIGddnew(x,sgm,k,theta,mu)))
    LL=-Lv
    return LL
def VGmom(x):
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    nu=3./(K-3.)
    sgm=np.sqrt(V*(K-3.)/3.)
    theta=S*np.sqrt(V)/3.
    mus=E-theta*nu
    return  abs(sgm),abs(nu),theta,mus
def VGmomdt(x,d):
    K=(scs.kurtosis(x)+3.0)*d
    S=scs.skew(x)*d
    V=np.var(x)*d
    E=np.mean(x)*d
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    nu=3./((K-3.))
    sgm=np.sqrt(V*(K-3.)/3.)
    theta=S*np.sqrt(V)/3.
    mus=E-theta*nu
    return  abs(sgm),abs(nu),theta,mus

def VGmomFull(x):
    d=1
    #T/n
    sgm0,nu0,theta0,mus0=VGmom(x)
    eps0=(theta0**2*nu0)/sgm0**2
    
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    feps=lambda eps:eps*(3+2*eps)**2/((1+4*eps+2*eps**2)*(1+eps))-2*S**2/K
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    epstar=newton(feps,eps0)
    sgm=np.sqrt(V/(1+epstar)/d)
    nu=K*d/3*((1+epstar)**2/(1+4*epstar+2*epstar**2))
    theta=S/(sgm**2*nu*d)*(1/(3+2*epstar))
    mus=E-theta*nu
    return abs(sgm),abs(nu),theta,mus



def VGddnew(x,sgm,nu,theta,mus):
    f=(np.sqrt(2)/(np.sqrt(np.pi)*sgm*sc.special.gamma(nu)))*(np.abs(x-mus)/np.sqrt(theta**2+2*sgm**2))**(nu-0.5)*np.exp((x-mus)*theta/sgm**2)*sc.special.kv(nu-0.5,np.abs(x-mus)*np.sqrt(theta**2+2*sgm**2)/sgm**2)
    return f
#@jit(nopython=False,parallel=True)
def loglikVGnum(pars,x):
    sgm=pars[0];nu=pars[1];theta=pars[2];mus=pars[3];
#mus=pars(1);
#mus=0;
    T=np.shape(x)[0];
    nL=-(0.5*T*np.log(2/np.pi)+np.sum((x-mus)*theta/sgm**2)-T*np.sum(np.log(sc.special.gamma(nu)*sgm))+np.sum(np.log(sc.special.kv(nu-0.5,(np.sqrt(2*sgm**2+theta**2)*np.abs(x-mus)/sgm**2))))+np.sum((nu-0.5)*(np.log(np.abs(x-mus)-0.5*np.log(2.*sgm**2+theta**2)))))
    return nL
def brloglikVG(pars,x):
    sgm=pars[0];nu=pars[1];theta=pars[2];mus=pars[3];
    Lv=np.sum(np.log(VGddnew(x,sgm,nu,theta,mus)))
    LL=-Lv
    return LL
@jit(nopython=True,parallel=True)
def appNormCDF(x):
    p=1.0/(np.exp(-358.0*x/23.0+111*np.arctan(37.0*x/294))+1.0)
    return p
@jit(nopython=True,parallel=False,nogil=True,fastmath=True)
def fastNpdf(x):
    C=1.0/np.sqrt(2.0*np.pi)
    return np.exp(-x*x*0.5)*C

def Pnormprodpdfbynchi2(z,mm1,mm2,ss1,ss2,rr,ispdf):
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
    nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l))-1)
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        
        out=Parallel(n_jobs=8)(delayed(lambda zz:quad(nrmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
        #quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
        out=quad(crmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
    return out

def normprodpdfbynchi2(z,mm1,mm2,ss1,ss2,rr,ispdf):
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
    nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l))-1)
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        
        out=quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        #
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
        out=quad(crmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
    return out

def nprodMLE(pars,ispdf,z,withmu,nocorr):
    mm1,mm2,ss1,ss2,rr=pars
    if nocorr:
        rr=0.0
    else:
        rr=1.0/(1.0+np.exp(-rr))
    if withmu:
        
        return -np.sum(np.log(vnormprodpdfbynchi2(z,mm1,mm2,abs(ss1),abs(ss2),rr,ispdf)))
    else:
        return -np.sum(np.log(NPCorr(z,0.0,0.0,abs(ss1),abs(ss2),rr)))
    

#%%
start='2023-03-05'
end='2024-03-05'
stocklist=['^GSPC']
yf.pdr_override()
for stock in stocklist:
    ps=pdr.get_data_yahoo(stock,start,end)
ps.Close.plot()
#sp=pd.read_clipboard()

x=np.diff(np.log(ps['Adj Close'].dropna().values))
#%%
mm1=1.0;ss1=2.0;mm2=0.5;ss2=2.0;rr=0.5
vnormprodpdfbynchi2=np.vectorize(normprodpdfbynchi2)
#normprodpdfbynchi2(0.05,mm1,mm2,ss1,ss2,rr)

#ps=pd.read_excel("C:/Users/aheki/Downloads/GIP_Agirlikli_Ortalama_Fiyat.xlsx")
#x=np.diff(np.log(ps.Close.dropna().values))
x0=VGmom(x)
#x0=VGmomFull(x)
xn0=NIGmom(x)
print("Check Error:%0.7f" %(loglikVGnum(x0,x)))
print("Check Error:%0.7f" %(brloglikVG(x0,x)))
print("Check Error:%0.7f" %(brloglikNIG(x0,x)))
optsCG = {'maxiter' : None, 'disp' : True,'gtol' : 1e-5,'norm' : np.inf, 'eps' : 1.4901161193847656e-07}
optsNM={'xtol': 1e-5,'ftol':1e-5,'disp':True,'maxfev':10000}
optcal=minimize(brloglikVG,xn0,args=(x),method='SLSQP',options=optsNM)
#%%
optcalNIG=minimize(brloglikNIG,xn0,args=(x),method='SLSQP',options=optsNM)
optparProd=minimize(fun=nprodMLE,x0=np.array([mm1,mm2,ss1,ss2,rr]),args=(True,x,True,False),method='SLSQP')
#%%
#fmin_bfgs(loglikVGnum,x0,args=(x))
#optVGp=sc.optimize.least_squares(loglikVGnum,x0,args=(x),xtol=1e-5, ftol=1e-5,verbose=1,max_nfev=10000)
#sgm,nu,theta,mu=optcal.x
sgmn,k,thetan,mun=optcalNIG.x
mm1,mm2,ss1,ss2,rr=optparProd.x
rr=1.0/(1.0+np.exp(-rr))
sx=np.sort(x)
#vgpdf=VGddnew(sx,sgm,nu,theta,mu)
NIGpdf=NIGddnew(sx,sgmn,k,thetan,mun)
Prodpdf=vnormprodpdfbynchi2(sx,mm1,mm2,abs(ss1),abs(ss2),rr,True)
sns.histplot(x,stat='density');plt.plot(sx,Prodpdf);plt.plot(sx,NIGpdf);
plt.title("Prod-NIG Fits Using Log Returns SPX (2023-2024)")
legends=[ps.columns[2]+':Prod '+start+' - '+end,ps.columns[2]+':NIG '+start+' - '+end]
plt.legend(legends)
#%%
%matplotlib qt
ecdf=ECDF(sx)
cdfmu=[simpson(Prodpdf[:t],sx[:t]) for t in np.linspace(1,sx.shape[0],sx.shape[0],dtype=int)]
cdfNIG=[simpson(NIGpdf[:t],sx[:t]) for t in np.linspace(1,sx.shape[0],sx.shape[0],dtype=int)]
plt.plot(sx,cdfmu,'--');plt.plot(sx,ecdf(sx));plt.plot(sx,cdfNIG,'--');plt.legend(['prod','emprical','NIG'])