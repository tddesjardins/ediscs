import photutils as phot
import numpy as np
from scipy import optimize
from astropy.stats import biweight_midvariance as bmv
import pyfits as pf
import pdb,os,math
import matplotlib.pyplot as plt
import warnings

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('ps',usedistiller='xpdf')
plt.close('all')

def getbinsize(min,max,space):

    r=max-min
    n=r/space

    return (min,max,n)

def gausswiki(x,a,b,c):

    return (a*np.exp((-1.0*(x-b)**2.0)/(2.0*(c**2.0))))

def gauss(x,a,b,c,d,e,f):
    """
    Form of the Gaussian function from the IDL function GAUSSFIT.
    Uses six parameters corresponding to:
    \ta = Height
    \tb = Center
    \tc = Sigma
    \td = Constant term
    \te = Linear term
    \tf = Quadratic term
    """
    
    z=(x-b)/c

    return (a*np.exp((-1.*(z**2.))/2.))+d+(e*x)+(f*(x**2.))

def gauss3(x,a,b,c):
    """
    Form of the Gaussian function from the IDL function GAUSSFIT.
    Uses three parameters corresponding to:
    \ta = Height
    \tb = Center
    \tc = Sigma
    """
    
    z=(x-b)/c

    return (a*np.exp((-1.*(z**2.))/2.))

def sigfunc(N,s,a,b):
    """
    Equation 3 from Labbe et al. (2003), ApJ, 125, 1107
    """
    
    return (N*s*(a+(b*N)))

def dophot(data,x,y,rad):

    positions=[]
    for i in range(len(x)):
        positions.append((x[i],y[i]))
    apertures=phot.CircularAperture(positions,r=rad)

    phot_table=phot.aperture_photometry(data,apertures)
    sum=phot_table['aperture_sum']

    return sum

def checkoverlap(x,y,flags,rad):
    """
    """

    for i in range(len(x)):
        for j in range(len(x)-i-1):
            j=j+i+1
            if flags[j] == 0:
                break
            dist=np.sqrt(((x[i]-x[j])**2.)+((y[i]-y[j])**2.))
            if dist <= rad:
                flags[j]=1
                break

    return flags
            
def main(image,segmap,niter=1200,pixscale=0.4,aprange=[0.5,3.0],level=5.0,
         border=500.0,nsamp=8,outplot='test.pdf',clname='',expmap='',persec=True,save='',maxrange=500.,rfactor=5.):

    if nsamp > 8:
        raise Warning('Keyword nsamp must be <= 8')

    if os.path.exists(outplot):
        os.remove(outplot)

    #Get image data and from that get the X- and Y-axis sizes of the image
    #The calls to the warnings module come from the PyFITS user manual and
    #supress the warnings from PyFITS concerning non-standard keywords in
    #DJ's image headers
    warnings.resetwarnings()
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    data=pf.open(image)[0].data
    detMap=pf.open(segmap)[0].data
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=UserWarning, append=True)

    (xs,ys)=(np.shape(data)[1],np.shape(data)[0])

    #Get exposure map values to make sure none of the regions fall on an
    #area of low (<90% of the median) exposure
    if expmap != '':
        exposure=pf.open(expmap)[0].data

    #Generate NITER randomly distributed (x,y) pairs no closer than BORDER
    #pixels from the edge of the image (for instances where there is empty
    #space along the edges of the images, i.e., mosaics).
    cenx=np.random.sample(size=niter)*(xs-(2.*border))+border
    ceny=np.random.sample(size=niter)*(ys-(2.*border))+border

    #Create apflag array where 0 (False) means the aperture is bad and
    #shouldn't be used (i.e., it contains a pixel above the detection level
    #or it is overlapping another aperture)
    apflag=np.ones(len(cenx))

    #Turn the aperture range into pixels, then check if the apertures are
    #overlapping at their maximum diameter
    aprange_pix=np.array(aprange)/pixscale
    maxrad=(aprange_pix[1])
    apflag=checkoverlap(cenx,ceny,apflag,maxrad)
    nonoverlap=np.where(apflag == 1.)[0]
    (newx,newy,flags)=(cenx[nonoverlap],ceny[nonoverlap],apflag[nonoverlap])

    #Check to be sure that the aperture is not on an area of low exposure
    if expmap != '':
        maxExp=np.max(exposure)
        for i in range(len(newx)):
            if exposure[newy[i],newx[i]] < (0.9*maxExp):
                flags[i]=0

    #Logarithmically space the aperture radii and then perform the photometry
    #Set up flags for the apertures (0 = aperture should not be used, 1 = aperture
    #is good)
    radii=np.logspace(np.log10(aprange_pix[0]),np.log10(aprange_pix[1]),nsamp)
    photometry=np.zeros((len(radii),len(newx)),dtype=np.float)
    maximum=np.zeros((len(radii),len(newx)),dtype=np.float)
    mean=np.zeros((len(radii),len(newx)),dtype=np.float)
    std=np.zeros((len(radii),len(newx)),dtype=np.float)
    for i in range(len(radii)):
        phot=dophot(data,newx,newy,radii[i])
        det=dophot(detMap,newx,newy,radii[i])
        photometry[i,:]=phot
        for j in range(len(cenx)):
            if flags[j] == 1.:
                if det[j] > 0.0:
                    flags[j]=0.
                if phot[j] == 0.:
                    flags[j]=0.
                if math.isnan(phot[j]) == True:
                    flags[j]=0.

    good=np.where(flags == 1.0)
    #Write out the aperture coordinates to a save file if
    #someone wants to see where the (good) apertures are
    #located. Good for verifying they aren't near sources
    #and adequately sample the whole image
    np.savetxt('test.cdt', np.c_[newx[good], newy[good]])

    #Make and plot the histograms, then fit each with a Gaussian
    (widths,heights)=([],[])
    colors=['black','blue','red','green','purple','cyan','magenta','yellow']
    spaces=np.zeros(nsamp)
    spaces[0:2]=0.1
    spaces[2:4]=0.1
    spaces[4:-1]=0.2
    spaces[-1]=0.3
    good=np.where(flags == 1.)
    plt.figure(1,figsize=(8.5,11))
    plt.subplot(211)

    for i in range(len(radii)):
        maxgood = maxrange
        goodi=np.where((flags == 1.) & (photometry[i,:] > -1.*maxgood) & (photometry[i,:] < maxgood))
        rms=bmv(photometry[i,goodi[0]])
        (bmin,bmax)=(-1.*rfactor*rms,rfactor*rms)
        if ((save != '') & (i == 0)):
            f=open(save,'w')
            for j in photometry[i,goodi[0]]:
                f.write(str(j)+'\n')
            f.close()
        res=getbinsize(bmin,bmax,spaces[i])
        binarr=np.linspace(np.floor(bmin),np.ceil(bmax),np.ceil(np.sqrt(len(photometry[i,good[0]]))))
        (n,bins,_)=plt.hist(photometry[i,good[0]],bins=binarr,
                            range=[bmin,bmax],histtype='step',color=colors[i])
        np.savetxt('photdata.dat', np.c_[photometry[i,good[0]]])
        heights.append(np.max(n))
        narr=np.array(n)
        barr=np.array(bins[:-1])+0.05
        gparguess=[np.max(narr),barr[np.where(narr == np.max(narr))[0][0]],np.std(barr)]
        (gfit,gcovariance)=optimize.curve_fit(gauss3,barr,narr,p0=gparguess,maxfev=100000000)
        if gfit[2] < 0.:
            gfit[2]=np.abs(gfit[2])
        xarr=np.round(np.linspace(bmin,bmax,20000),4)
        yarr=gauss3(xarr,gfit[0],gfit[1],gfit[2])
        plt.plot(xarr,yarr,'--',color=colors[i],label=r'$N = '+str(np.round(np.sqrt(np.pi*(radii[i]**2.0)),1))+'$, $\sigma = '+str(np.round(gfit[2],2))+'$')

        widths.append(gfit[2])

    (plmin,plmax)=(bmin,bmax)
    plt.axis([plmin,plmax,0,np.ceil(np.max(heights)/10.)*10.+10])
    if clname != '':
        plt.annotate('Cluster: '+clname.strip()+'\nTotal Apertures: '+str(len(good[0])).strip(),xy=(bmax*0.3,np.max(narr)*0.75),
                     xytext=(bmax*0.3,np.max(narr)*0.75),color='black',weight='heavy')
    else:
        plt.annotate('Total Apertures: '+str(len(good[0])).strip(),xy=(bmax*0.3,np.max(narr)*0.75),
                     xytext=(bmax*0.3,np.max(narr)*0.75),color='black',weight='heavy')
    plt.xlabel('Sum in Aperture of Size N pixels / counts sec'+r'$^{-1}$',weight='heavy')
    plt.ylabel('Number of Apertures',weight='heavy')
    plt.legend(loc=2,handlelength=3)

    plt.subplot(212)
    
    #Fit the resulting sigma(N) data using the sigfunc() function
    radn=np.sqrt(np.pi*(radii**2.0))
    (sfit,scovariance)=optimize.curve_fit(sigfunc,radn,widths)
    xarr=np.linspace(0.,30.,20000)
    yarr=sigfunc(xarr,sfit[0],sfit[1],sfit[2])
    sigfit=plt.plot(xarr,yarr,'k--',label='Fit')
    data=plt.plot(radn,widths,'ro',markersize=8,label='Data')
    plt.axis([0.,np.ceil(np.max(radn))+2,0.,np.ceil(np.max(widths))+1.0])
    eqnstring=r'$\sigma='+str(round(sfit[0],2)).strip()+r'N\times['+str(round(sfit[1],2)).strip()+r'+('+str(round(sfit[2],2)).strip()+\
                 r'N)]$'
    plt.annotate(eqnstring,xy=(1.0,np.max(widths)*0.75),xytext=(1.0,np.max(widths)*0.75),color='black',weight='heavy',fontsize=20)
    plt.xlabel('Linear Aperture Size N / pixels',weight='heavy')
    plt.ylabel(r'$\sigma$ / counts sec$^{-1}$',weight='heavy')
    plt.legend(handlelength=3,numpoints=1,loc=2)

    plt.tight_layout()
    plt.savefig(outplot,format='pdf',dpi=6000)
    plt.close('all')

    return(sfit[0],sfit[1],sfit[2])
