import numpy as np
from statsmodels.formula.api import ols,rlm
import statsmodels.api as sm
from astropy import coordinates
from astropy.table import Table
from astropy import units as u
import pdb, shlex, os, shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import photcheck as pc
import subprocess as sp
import labbe_depth as lb
from astropysics import obstools as obs
import matplotlib.pyplot as plt
import pyfits as pf
from astropy.stats.funcs import biweight_location as bl
from scipy.optimize import curve_fit

#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rc('ps',usedistiller='xpdf')

#-----------------------------------
# GENERAL NOTES
#
# Required files:
#    Photometry from SExtractor for the WFI, MOSAIC, and
#       NEWFIRM observations
#
#    Dennis Just's "VLT.starflag.dat" file for each cluster
#
#    FORS photometry to solve for WFI zeropoints
#-----------------------------------

#Constants
vega2AB={'bctio':-0.09949,'bkpno':-0.10712,'v':0.01850,'r':0.19895,'i':0.42143,'k':1.84244}
medCterms={'VRv':-0.151,'VRr':0.0245,'VIv':-0.0725,'VIi':0.1465,'RIr':0.015,'RIi':0.238}

#-----------------------------------
def fixedVRv(x,zp):
    return (medCterms['VRv']*x)+zp

def fixedVRr(x,zp):
    return (medCterms['VRr']*x)+zp

def fixedVIv(x,zp):
    return (medCterms['VIv']*x)+zp

def fixedVIi(x,zp):
    return (medCterms['VIi']*x)+zp

def fixedRIr(x,zp):
    return (medCterms['RIr']*x)+zp

def fixedRIi(x,zp):
    return (medCterms['RIi']*x)+zp

#-----------------------------------
def updatecat(catalog,fieldName,outname,megacat='/Volumes/BAHAMUT/megacat.v5.7.fits'):

    hdu = pf.open(catalog)
    oldData = hdu[1].data
    field=[fieldName for x in range(len(oldData))]

    (filters, apertures) = (['B','V','R','I','z','K'], ['1','2','3','iso','auto'])
    for filt in filters:
        for ap in apertures:
            fluxes = 'f'+filt+ap
            bad = np.where(np.log10(np.abs(oldData[fluxes])) < -6.)
            oldData[fluxes][bad] = -77

    (zLDP,Q)=(np.zeros(len(oldData))-99,np.zeros(len(oldData))-99)
    
    megatab=pf.open('/Users/tyler/megacat.v5.7.fits')
    megadat=megatab[1].data
    (mzldp,mq,megaRA,megaDec)=(megadat['zldp'],megadat['q'],megadat['RA'],megadat['DEC'])
        
    wfiSky=coordinates.SkyCoord(ra=oldData['ra']*u.degree, dec=oldData['dec']*u.degree)
    megaSky=coordinates.SkyCoord(ra=megaRA*u.degree, dec=megaDec*u.degree)
    (idx,d2d,_)=wfiSky.match_to_catalog_sky(megaSky)
    match=np.where(d2d.arcsec < 0.5)
    zLDP[match]=mzldp[idx][match]
    Q[match]=mq[idx][match]
    
    final={'field':field,'ids':oldData['ids'],'ra':oldData['ra'],'dec':oldData['dec'],'x':oldData['x'],'y':oldData['y'],'ebv':oldData['ebv'],
           'fwhmR':oldData['fwhmR'],
           'fB1':oldData['fB1'],'fB1err':oldData['fB1err'],'fB2':oldData['fB2'],'fB2err':oldData['fB2err'],'fB3':oldData['fB3'],'fB3err':oldData['fB3err'],
           'fBiso':oldData['fBiso'],'fBisoerr':oldData['fBisoerr'],'fBauto':oldData['fBauto'],'fBautoerr':oldData['fBautoerr'],'fV1':oldData['fV1'],
           'fV1err':oldData['fV1err'],'fV2':oldData['fV2'],'fV2err':oldData['fV2err'],'fV3':oldData['fV3'],'fV3err':oldData['fV3err'],'fViso':oldData['fViso'],
           'fVisoerr':oldData['fVisoerr'],'fVauto':oldData['fVauto'],'fVautoerr':oldData['fVautoerr'],'fR1':oldData['fR1'],'fR1err':oldData['fR1err'],
           'fR2':oldData['fR2'],'fR2err':oldData['fR2err'],'fR3':oldData['fR3'],'fR3err':oldData['fR3err'],'fRiso':oldData['fRiso'],'fRisoerr':oldData['fRisoerr']
           ,'fRauto':oldData['fRauto'],'fRautoerr':oldData['fRautoerr'],'fI1':oldData['fI1'],'fI1err':oldData['fI1err'],'fI2':oldData['fI2'],
           'fI2err':oldData['fI2err'],'fI3':oldData['fI3'],'fI3err':oldData['fI3err'],'fIiso':oldData['fIiso'],'fIisoerr':oldData['fIisoerr'],
           'fIauto':oldData['fIauto'],'fIautoerr':oldData['fIautoerr'],'fz1':oldData['fz1'],'fz1err':oldData['fz1err'],'fz2':oldData['fz2'],
           'fz2err':oldData['fz2err'],'fz3':oldData['fz3'],'fz3err':oldData['fz3err'],'fziso':oldData['fziso'],'fzisoerr':oldData['fzisoerr'],
           'fzauto':oldData['fzauto'],'fzautoerr':oldData['fzautoerr'],'fK1':oldData['fK1'],'fK1err':oldData['fK1err'],'fK2':oldData['fK2'],
           'fK2err':oldData['fK2err'],'fK3':oldData['fK3'],'fK3err':oldData['fK3err'],'fKiso':oldData['fKiso'],'fKisoerr':oldData['fKisoerr'],
           'fKauto':oldData['fKauto'],'fKautoerr':oldData['fKautoerr'],'zLDP':zLDP,'Q':Q,'starB':oldData['starB'],'starV':oldData['starV'],
           'starR':oldData['starR'],
           'starI':oldData['starI'],'starz':oldData['starz'],'starK':oldData['starK'],'sexflagB':oldData['sexflagB'],'sexflagV':oldData['sexflagV'],
           'sexflagR':oldData['sexflagR'],'sexflagI':oldData['sexflagI'],'sexflagz':oldData['sexflagz'],'sexflagK':oldData['sexflagK']}

    tab = Table(final, names=('field','ids','ra','dec','x','y','ebv','fwhmR','fB1','fB1err','fB2',
                              'fB2err','fB3','fB3err','fBiso','fBisoerr',
                             'fBauto','fBautoerr','fV1','fV1err','fV2','fV2err','fV3','fV3err','fViso','fVisoerr','fVauto',
                             'fVautoerr','fR1','fR1err','fR2','fR2err','fR3','fR3err','fRiso','fRisoerr','fRauto',
                             'fRautoerr','fI1','fI1err','fI2','fI2err','fI3','fI3err','fIiso','fIisoerr','fIauto',
                             'fIautoerr','fz1','fz1err','fz2','fz2err','fz3','fz3err','fziso','fzisoerr','fzauto',
                             'fzautoerr','fK1','fK1err','fK2','fK2err','fK3','fK3err','fKiso','fKisoerr','fKauto',
                             'fKautoerr','zLDP','Q','starB','starV','starR','starI','starz','starK','sexflagB','sexflagV','sexflagR','sexflagI',
                             'sexflagz','sexflagK'))
    tab.write(outname, format='fits', overwrite=True)

#-----------------------------------
def matchxy(x1,y1,x2,y2,tol=0.1):

    match=[]
    for i in range(len(x1)):
        cdt=(x1[i],y1[i])
        dist=np.sqrt((cdt[0]-x2)**2.+(cdt[1]-y2)**2.)
        if np.min(dist) <= tol:
            match.append(np.where(dist == np.min(dist))[0][0])

    return np.array(match)

#-----------------------------------
def backZP(flux,mag):
    """PURPOSE: Backout the zeropoint for a source knowing its flux in detector units and its physical magnitude

    INPUTS:
    \tflux - flux in detector units (counts or counts/second)
    \tmag  - magnitude in physical units (note that if this is in AB mag, the result includes the AB conversion factor)
    
    RETURNS: The zeropoint to convert between detector flux and physical magnitude
    """

    return 2.5*np.log10(flux)+mag

#-----------------------------------
def getSmoothFactor(rcat,xcat,class_star=0.0,border=1500.,pixscale=0.238,save=False):

    rangeMagR=[-15.,-12.]
    rangeMagX=[-10.,-5.]

    (rflux,rfwhm,starR)=np.loadtxt(rcat,usecols=(6,9,14),unpack=True,comments='#')
    (xcdt,ycdt,xflux,xfwhm,starX)=np.loadtxt(xcat,usecols=(2,3,6,9,14),unpack=True,comments='#')

    (rmags,xmags)=(-2.5*np.log10(rflux),-2.5*np.log10(xflux))
    goodX=np.where((xfwhm > 0.) & (xcdt < np.max(xcdt)-border) &
                  (xcdt > np.min(xcdt)+border) & (ycdt < np.max(ycdt)-border) &
                  (ycdt > np.min(ycdt)+border))
    goodR=np.where(starR >= class_star)

    plt.scatter(rmags,rfwhm*3600.,alpha=0.05,color='r')
    plt.scatter(xmags[goodX],xfwhm[goodX]*3600.,alpha=0.05,color='b')

    (rfwhm,xfwhm)=(rfwhm*3600.,xfwhm*3600.)
    (rfwhmSub,xfwhmSub)=(rfwhm[np.where((rmags >= rangeMagR[0]) & (rmags <= rangeMagR[1]) & (starR >= class_star))],
                         xfwhm[np.where((xmags >= rangeMagX[0]) & (xmags <= rangeMagX[1]) & (starX >= class_star) & (xcdt < np.max(xcdt)-border) &
                  (xcdt > np.min(xcdt)+border) & (ycdt < np.max(ycdt)-border) &
                  (ycdt > np.min(ycdt)+border))])

    (avgR,avgX)=(bl(rfwhmSub),bl(xfwhmSub))

    xx=[-100,100]
    yy1=[avgR,avgR]
    yy2=[avgX,avgX]
    plt.plot(xx,yy1,'k-')
    plt.plot(xx,yy2,'k--')
    plt.axis([-20,-1,0.5,15])

    (pR,pX)=(avgR/pixscale,avgX/pixscale)
    sig=np.sqrt((pX**2.)-(pR**2.))/2.355
    print 'Avg. R = '+str(avgR)+' arcsec\nAvg. X = '+str(avgX)+' arcsec\nSigma smooth factor: '+str(sig)+' pixels'

    if save == True:
        plt.savefig('seeing_comp.pdf',format='pdf',dpi=6000.)
    else:
        plt.show()

#-----------------------------------
def addquad(xerr,orerr,nrerr,xflux,orflux,nrflux):

    value=(xflux/nrflux)*orflux

    return value*np.sqrt((xerr/xflux)**2. + (nrerr/nrflux)**2. + (orerr/orflux)**2.)
    
    
#-----------------------------------
def photscript(listfile,clname,photfile='photscript'):
    """
    PURPOSE: Generate a script (to be called on with source from the command line) that will run SExtractor on
    \tall input images in dual-image mode (using the R-band for detection) for a cluster.

    INPUTS:
    \tlistfile -

    """

    (keys,files)=np.loadtxt(listfile,usecols=(0,1),unpack=True,dtype={'names':('keys','files'), 'formats':('S4','S30')})
    imgs={}
    for x in range(len(keys)):
        imgs[keys[x]]=files[x]
        
    outfile=open('photscript','w')
    
    string1='sex -c wfi_tyler.sex -BACKPHOTO_TYPE "GLOBAL" -CATALOG_NAME '
    string2=' -CHECKIMAGE_TYPE "-BACKGROUND" -CHECKIMAGE_NAME '
    finalString=''

    finalString=finalString+string1+clname+'_r.cat'+' -CHECKIMAGE_TYPE "-BACKGROUND,SEGMENTATION" -CHECKIMAGE_NAME "' \
        +imgs['rimg'][:-5]+'_bkgsub.fits'+','+imgs['rimg'][:-5]+'_segmap.fits" '+imgs['rimg']+','+imgs['rimg']+'\n'
        
    if 'bimg' in keys:
        finalString=finalString+string1+clname+'_b.cat'+string2+imgs['bimg'][:-5]+'_bkgsub.fits '+imgs['rimg']+','+imgs['bimg']+'\n'
    if 'vimg' in keys:
        finalString=finalString+string1+clname+'_v.cat'+string2+imgs['vimg'][:-5]+'_bkgsub.fits '+imgs['rimg']+','+imgs['vimg']+'\n'
    if 'iimg' in keys:
        finalString=finalString+string1+clname+'_i.cat'+string2+imgs['iimg'][:-5]+'_bkgsub.fits '+imgs['rimg']+','+imgs['iimg']+'\n'
    if 'zimg' in keys:
        finalString=finalString+string1+clname+'_z.cat'+string2+imgs['zimg'][:-5]+'_bkgsub.fits '+imgs['rimg']+','+imgs['zimg']+'\n'
    if 'kimg' in keys:
        finalString=finalString+string1+clname+'_k.cat'+string2+imgs['kimg'][:-5]+'_bkgsub.fits '+imgs['rimg']+','+imgs['kimg']+'\n'

    out=open(photfile,'w')
    out.write(finalString)
    out.close()
        
#-----------------------------------
def flags(flux,err,x,y,xmin=0,xmax=1e6,ymin=0,ymax=1e6):
    
    for j in range(len(flux)):
        if flux[j] == -99.0:
            err[j] = -99.0
        if np.log10(np.abs(flux[j])) > 4.0:
            flux[j] = -77.0
            err[j] = -77.0
        if np.log10(np.abs(flux[j])) < -8.0:
            flux[j] = -66.0
            err[j] = -66.0
        if x[j] < xmin:
            flux[j] = -55.0
            err[j] = -55.0
        if x[j] > xmax:
            flux[j] = -55.0
            err[j] = -55.0
        if y[j] < ymin:
            flux[j] = -55.0
            err[j] = -55.0
        if y[j] > ymax:
            flux[j] = -55.0
            err[j] = -55.0

    return flux,err

#-----------------------------------
def fixData(data,flag=-88.0):

    keys=data.keys()
    for x in keys:
        if x != 'ids':
            nan=np.where(np.isnan(data[x]))
            inf=np.where(np.isinf(data[x]))
            data[x][nan]=flag
            data[x][inf]=flag

    return data

#-----------------------------------
def sigfunc(N,s,a,b):
    """
    PURPOSE: Function from qquation 3 from Labbe et al. (2003), ApJ, 125, 1107. 
    """
    
    return (N*s*(a+(b*N)))

#-----------------------------------
def binAvgData(x, y, npts, minimum = -999999.0, maximum = 9999999.0):

    use = np.where((x > minimum) & (x < maximum))
    x, y = x[use], y[use]
    sort = np.argsort(x)
    xs, ys = x[sort], y[sort]
    xm, ym = np.array([]), np.array([])
    i = 0
    for j in range(int(np.ceil(len(x)/npts))):
        if i+npts <= len(x):
            xm = np.append(xm, np.mean(xs[i:i+npts]))
            ym = np.append(ym, np.mean(ys[i:i+npts]))
        else:
            xm = np.append(xm, np.mean(xs[i:]))
            ym = np.append(ym, np.mean(ys[i:]))
        i = i + npts

    return (xm, ym)

#-----------------------------------
def binAvgDataFixed(x, y, width, minimum = -999999.0, maximum = 9999999.0):

    use = np.where((x > minimum) & (x < maximum))
    x, y = x[use], y[use]
    sort = np.argsort(x)
    xs, ys = x[sort], y[sort]
    xm, ym = np.array([]), np.array([])
    i = np.min(x)
    while i + width < np.max(x):
        xm = np.append(xm, np.mean(xs[np.where((xs > i) & (xs <= i+width))]))
        ym = np.append(ym, np.mean(ys[np.where((xs > i) & (xs <= i+width))]))
        i = i + width

    return (xm, ym)

#-----------------------------------
def getEBV(ra,dec):
    """
    PURPOSE: For a given RA and Dec, lookup the E(B-V) value from the Schlegel dust maps.

    INPUTS:
    \tra  - Right ascension (deg; J2000)
    \tdec - Declination (deg; J2000)

    RETURNS: E(B-V)
    """

    if os.path.exists('SFD_dust_4096_ngp.fits') == False:
        shutil.copy('/Users/tyler/Downloads/SFD_dust_4096_ngp.fits','.')
    if os.path.exists('SFD_dust_4096_sgp.fits') == False:
        shutil.copy('/Users/tyler/Downloads/SFD_dust_4096_sgp.fits','.')

    galcoord = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5').galactic
    (latitude, longitude) = (galcoord.l.degree, galcoord.b.degree)
    ebv = obs.get_SFD_dust(latitude, longitude)
        
    return ebv

#-----------------------------------
def calflux(flux, zp, abconv = 0.0):
    """
    PURPOSE: Converts an instrumental flux into a physical one in flux space to preserve negative values.

    INPUTS:
    \tflux   - instrumental flux
    \tzp     - zeropoint to convert instrumental to calibrated flux
    \tabconv - conversion factor from Vega to AB magnitudes (optional)

    RETURNS: Flux in uJy
    """

    return ((flux)/(10.0**(0.4*(zp + abconv)))) * 3631.0 * 1e6
        
#-----------------------------------
def flux2mag(flux, zp, abconv = 0.0):

    return -2.5 * np.log10(flux) + zp + abconv

#-----------------------------------
def mag2flux(mag):

    return 10.0**(-0.4 * mag)

#-----------------------------------
def ab2ujy(mag):

    jy = 10.0**(-0.4 * mag) * 3631.0
    
    return jy * 1e6

#-----------------------------------
def ujy2abmag(flux):
    
    return -2.5 * np.log10(flux/1e6) + 8.9

#-----------------------------------
def seeingCorr(rr, rm, mm, outfile='seeingcorr.dat'):
    
    res=(mm*rr)/rm
    out=open(outfile,'w')
    for x in range(len(mm)):
        out.write(str(mm[x])+'\t'+str(res[x])+'\n')
    out.close()
    return res

#-----------------------------------
def zpWFI(ra, dec, v, r, i, v3, r3, i3, starR, photref = 'fors.dat', starref = 'VLT.starflag.dat', tol=0.01, synth=False):
    """
    Notes: Adapted from Dennis Just's "bvriz.pro" IDL code
    """
    
    (wmin, starflag) = np.loadtxt('VLT.starflag.dat', usecols=(0, 1), unpack = True, comments = '#')
    (forsRA, forsDec, forsV, forsR, forsI) = np.loadtxt('fors.dat', usecols = (1, 2, 9, 10, 11),
                                                        unpack = True, comments = '#')
     
    #Identify stars based on Dennis's criteria. Create new arrays of only the coordinates
    #and photometry of the stars from the FORS and WFI data. Convert the FORS RA from hours
    #to degrees for matching. (starflag = 0 = galaxies, 1 = stars)
    stars_fors = np.where((wmin > 0.2) & (starflag == 1) & (forsR <= 22.0) & (forsR > 16.5)
                          & (forsV > 16.5) & (forsI > 16.5))

    rinst=-2.5*np.log10(r)
    star_wfi = np.where((starR >= 0.0))# & (rinst >= -15.) & (rinst <= -12.))
    (sforsRA, sforsDec, sforsV, sforsR, sforsI) = (forsRA[stars_fors], forsDec[stars_fors],
                                                   forsV[stars_fors], forsR[stars_fors],
                                                   forsI[stars_fors])
    (swfiRA, swfiDec, swfiV, swfiR, swfiI) = (ra[star_wfi], dec[star_wfi], v3[star_wfi],
                                              r3[star_wfi], i3[star_wfi])
    sforsRAh = sforsRA*15.0

    #Match coords here!
    forsCat=coordinates.SkyCoord(ra=sforsRAh*u.degree, dec=sforsDec*u.degree)
    wfiCat=coordinates.SkyCoord(ra=swfiRA*u.degree, dec=swfiDec*u.degree)
    idx, d2d, _ = wfiCat.match_to_catalog_sky(forsCat)
    match = np.where(d2d.arcsec <= 1.0)

#    pdb.set_trace()

    (sforsVmatch,sforsRmatch,sforsImatch)=(sforsV[idx][match], sforsR[idx][match],
                                           sforsI[idx][match])

    #Create initial guesses of the zeropoints, then fit the zeropoints to the
    #(V-R) and (R-I) colors using an orthogonal least squares (OLS) regression
    #to protect against outliers.
    (zpV, zpR, zpI) = (sforsVmatch + (2.5 * np.log10(swfiV[match])),
                       sforsRmatch + (2.5 * np.log10(swfiR[match])),
                       sforsImatch + (2.5 * np.log10(swfiI[match])))

    vicolor=sforsVmatch-sforsImatch

    if synth == True:
        #Try correcting synthetic R-band magnitudes
        (corrslope,corrint)=(0.0133025347383,-0.00337599439368)
        corrR = (vicolor*corrslope) + corrint
        sforsRmatch = sforsRmatch - corrR

    vrcolor=sforsVmatch-sforsRmatch
    ricolor=sforsRmatch-sforsImatch

    #For the V-R color, exclude stars with V-R >= 0.7 for clusters with synthesized R-band magnitudes
    (vrrange,virange,rirange)=(np.where((vrcolor > 0.) & (vrcolor < 2.5) & (zpV > 22.5) & (zpR > 22.5) & (zpI > 22.5)),
                               np.where((vicolor > 0.) & (vicolor < 2.5) & (zpV > 22.5) & (zpR > 22.5) & (zpI > 22.5)),
                               np.where((ricolor > 0.) & (ricolor < 2.5) & (zpV > 22.5) & (zpR > 22.5) & (zpI > 22.5)))
    
    if synth == True:
        goodvr = np.where((vrcolor > 0.0) & (vrcolor < 2.5))
        (zpVvr, zpRvr, vr) = (zpV[goodvr], zpR[goodvr], vrcolor[goodvr])
        vfit0=curve_fit(fixedVRv,vr,zpVvr)[0][0]
        rfit0=curve_fit(fixedVRr,vr,zpRvr)[0][0]
#        (vfit0,vfit1)=rlm('zpVvr~vr',data=dict(y=zpVvr,x=vr),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000,tol=1e-10).params
#        (rfit0,rfit1)=rlm('zpRvr~vr',data=dict(y=zpRvr,x=vr),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000,tol=1e-10).params
    else:
        goodvr = np.where((vrcolor > 0.) & (vrcolor < 2.5))
        (zpVvr, zpRvr, vr) = (zpV[goodvr], zpR[goodvr], vrcolor[goodvr])
        vfit0=curve_fit(fixedVRv,vr,zpVvr)[0][0]
        rfit0=curve_fit(fixedVRr,vr,zpRvr)[0][0]
#        (vfit0,vfit1)=rlm('zpVvr~vr',data=dict(y=zpVvr,x=vr),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000,tol=1e-10).params
#        (rfit0,rfit1)=rlm('zpRvr~vr',data=dict(y=zpRvr,x=vr),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000,tol=1e-10).params
         
    (zpVvi, zpVvr, zpRri, zpRvr, zpIri, zpIvi) = (zpV[virange], zpV[vrrange], zpR[rirange], zpR[vrrange], zpI[rirange], zpI[virange])
    (vr,ri,vi) = (vrcolor[vrrange],ricolor[rirange],vicolor[virange])
     
#    (vfit20,vfit21)=rlm('zpVvi~vi',data=dict(y=zpVvi,x=vi),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000).params
#    (rfit20,rfit21)=rlm('zpRri~ri',data=dict(y=zpRri,x=ri),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000).params
#    (ifit0,ifit1)=rlm('zpIvi~vi',data=dict(y=zpIvi,x=vi),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000).params
#    (ifit20,ifit21)=rlm('zpIri~ri',data=dict(y=zpIri,x=ri),M=sm.robust.norms.TukeyBiweight()).fit(conv='sresid',maxiter=1000).params
    vfit20=curve_fit(fixedVIv,vi,zpVvi)[0][0]
    rfit20=curve_fit(fixedRIr,ri,zpRri)[0][0]
    ifit0=curve_fit(fixedVIi,vi,zpIvi)[0][0]
    ifit20=curve_fit(fixedRIi,ri,zpIri)[0][0]

    #Save color terms and zeropoints to output file
#    zpfile=open('wfi_zeropoints.dat','w')
#    cnames=['VR(V)','VR(R)','VI(V)','VI(I)','RI(R)','RI(I)']
#    cterms=[str(np.around(vfit1,3)),str(np.around(rfit1,3)),str(np.around(vfit21,3)),str(np.around(ifit1,3)),
#            str(np.around(rfit21,3)),str(np.around(ifit21,3))]
#    czeros=[str(np.around(vfit0,3)),str(np.around(rfit0,3)),str(np.around(vfit20,3)),str(np.around(ifit0,3)),
#            str(np.around(rfit20,3)),str(np.around(ifit20,3))]
#    zpfile.write('#COLOR\tSLOPE\tZEROPOINT\n')
#    for kk in range(len(cnames)):
#        zpfile.write(cnames[kk]+'\t'+cterms[kk]+'\t'+czeros[kk]+'\n')
#    zpfile.close()
     
#    print sforsRAh[idx][match][np.where(zpR - (rfit0 + (rfit1 * vrcolor)) > 0.1)]
#    print sforsDec[idx][match][np.where(zpR - (rfit0 + (rfit1 * vrcolor)) > 0.1)]

#    pdb.set_trace()
    plt.plot(vicolor,zpI,'ro')
    plt.plot(vicolor,zpV,'bo')
    plt.plot(vrcolor,zpR,'ko')
    xx=np.array([-100.0,100.0])
    yy=ifit0+(xx*medCterms['VIi'])
    yy2=vfit20+(xx*medCterms['VIv'])
    yy3=rfit0+(xx*medCterms['VRr'])
    plt.plot(xx,yy,'r--')
    plt.plot(xx,yy2,'b--')
    plt.plot(xx,yy3,'k--')
    plt.axis([0,2.5,np.min([ifit0,vfit20,rfit0])-0.5,np.max([ifit0,vfit20,rfit0])+1.])
    plt.xlabel('(V-I)')
    plt.ylabel('ZP')
    plt.savefig('slopecheck.pdf', format='pdf', dpi=6000)
    plt.close()

#    pdb.set_trace()

#    print (vfit1, vfit21, rfit1, ifit1, rfit21, ifit21)

#    pdb.set_trace()

    #Iterate over the WFI photometry (used 20 iterations from Dennis's code) to
    #sovle for the zeropoint for each source. When there is no V-band data, use
    #the (R-I) color (and the assosciated fit), otherwise use the (V-R) and (V-I)
    #colors.
    (nv,nr,ni)=(0,0,0)
    (zpV0,zpR0,zpI0)=(vfit0+(0.5*medCterms['VRv']), rfit0+(0.5*medCterms['VRr']), ifit0+(0.5*medCterms['VIi']))
    (vmag,rmag,imag)=(np.empty(len(starR)),np.empty(len(starR)),np.empty(len(starR)))
    (zpv,zpr,zpi)=(np.empty(len(starR)),np.empty(len(starR)),np.empty(len(starR)))
    rpflag=np.empty(len(starR))
    for j in range(len(starR)):
        (zpv[j], zpr[j], zpi[j]) = (zpV0, zpR0, zpI0)
        (ii, rdiff, vdiff, idiff) = (0, 1.0, 1.0, 1.0)

        if v[j] > 0:
            (zpcomp,compmag) = (zpV0,flux2mag(v[j],zpV0))
        else:
            (zpcomp,compmag) = (zpI0,flux2mag(i[j],zpI0))
        oldr=flux2mag(r[j],zpr[j])
        rmag[j]=oldr
        while (ii <= 20) and (np.abs(rdiff) > tol):
            if v[j] > 0:
                rpflag[j]=ii
                zpr[j]=medCterms['VRr']*(compmag-rmag[j])+rfit0
                zpcomp=medCterms['VRv']*(compmag-rmag[j])+vfit0
                rmag[j]=flux2mag(r[j],zpr[j])
                compmag=flux2mag(v[j],zpcomp)
            elif ((v[j] <= 0) & (i[j] > 0)):
                rpflag[j]=-1
                zpr[j]=medCterms['RIr']*(rmag[j]-compmag)+rfit20
                zpcomp=medCterms['RIi']*(rmag[j]-compmag)+ifit20
                rmag[j]=flux2mag(r[j],zpr[j])
                compmag=flux2mag(i[j],zpcomp)
            else:
                rpflag[j]=-1
                rmag[j]=flux2mag(r[j],zpr[j])
            rdiff=oldr-rmag[j]
            oldr=rmag[j]
            ii += 1
#            if ii == 2:
#                pdb.set_trace()
#        print j,ii-1,rdiff,'('+str(tol)+')'
        ii = 0

        if v[j] > 0:
            oldv=flux2mag(v[j],zpv[j])
            vmag[j]=oldv
            compmag=rmag[j]
            while (ii <= 20) and (np.abs(vdiff) > tol):
                zpv[j]=medCterms['VRv']*(vmag[j]-compmag)+vfit0
                vmag[j]=flux2mag(v[j],zpv[j])
                vdiff=oldv-vmag[j]
                oldv=vmag[j]
                ii += 1
        else:
            vmag[j]=-99
        ii = 0

        if i[j] > 0:
            oldi=flux2mag(i[j],zpi[j])
            imag[j]=oldi
            if v[j] > 0:
                (fit0, fit1, compmag, cfit0, cfit1) = (ifit0, medCterms['VIi'], vmag[j], vfit20, medCterms['VIv'])
            else:
                (fit0, fit1, compmag, cfit0, cfit1) = (ifit20, medCterms['RIi'], rmag[j], rfit20, medCterms['RIr'])
            while (ii <= 20) and (np.abs(idiff) > tol):
                zpi[j]=fit1*(compmag-imag[j])+fit0
                imag[j]=flux2mag(i[j],zpi[j])
                idiff=oldi-imag[j]
                oldi=imag[j]
                ii += 1
        else:
            imag[j]=-99
            
    #Convert the Vega magnitudes to AB magnitudes and uJy fluxes. Return the uJy fluxes
    #and the AB magnitude zeropoints.

#    print '\nNumber of sources needing more than 10 iterations for ZP:\n\tR-band ('+str(nr)\
#        +')\n\tV-band ('+str(nv)+')\n\tI-band ('+str(ni)+')\n'
    
    (vmagab, rmagab, imagab) = (vmag + vega2AB['v'], rmag + vega2AB['r'], imag + vega2AB['i'])
    (vab, rab, iab) = (ab2ujy(vmagab), ab2ujy(rmagab), ab2ujy(imagab))
    return (vab, rab, iab, zpv + vega2AB['v'], zpr + vega2AB['r'], zpi + vega2AB['i'],rpflag)
                
#-----------------------------------
def mergeLists(b="",v="",r="",i="",z="",k="",rimg="",zpb=0.0,zpk=0.0,null=-99):
    """
    """

    (foo1,foo2)=np.loadtxt(r, usecols = (0,1), unpack=True, comments= '#')
    nsrcs=len(foo1)
    nullArr = np.zeros(nsrcs) + null
    columnsR = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    columnsX = (4, 5, 6, 7, 8, 9, 14, 15)
    
    if b != "":
        (b1, b2, b3, biso, bauto, fwhmB, starB, sexflagB) = np.loadtxt(b, usecols = columnsX,
                                                                       unpack = True, comments = '#')
        fwhmB = fwhmB * 3600.0
    else:
        (b1, b2, b3, biso, bauto, starB, fwhmB, sexflagB) = (nullArr, nullArr, nullArr, nullArr, nullArr, nullArr,
                                                             nullArr, nullArr)
         
    if v != "":
        (v1, v2, v3, viso, vauto, fwhmV, starV, sexflagV) = np.loadtxt(v, usecols = columnsX, unpack = True, comments = '#')
        fwhmV = fwhmV * 3600.0
    else:
        (v1, v2, v3, viso, vauto, fwhmV, starV, sexflagV) = (nullArr, nullArr, nullArr, nullArr, nullArr, nullArr,
                                                             nullArr, nullArr)

    (x, y, r1, r2, r3, riso, rauto, fwhmR, kronR, aimage,
     bimage, isoarea, starR, sexflagR) = np.loadtxt(r, usecols = columnsR, unpack = True, comments = '#')
    fwhmR = fwhmR * 3600.0
    majorax = kronR * aimage
    minorax = kronR * bimage
    
    if i != "":
        (i1, i2, i3, iiso, iauto, fwhmI, starI, sexflagI) = np.loadtxt(i, usecols = columnsX, unpack = True, comments = '#')
        fwhmI = fwhmI * 3600.0
    else:
        (i1, i2, i3, iiso, iauto, fwhmI, starI, sexflagI) = (nullArr, nullArr, nullArr, nullArr, nullArr, nullArr,
                                                             nullArr, nullArr)
    
    if z != "":
        (z1, z2, z3, ziso, zauto, fwhmz, starz, sexflagz) = np.loadtxt(z, usecols = columnsX, unpack = True, comments = '#')
        fwhmz = fwhmz * 3600.0
    else:
        (z1, z2, z3, ziso, zauto, fwhmz, starz, sexflagz) = (nullArr, nullArr, nullArr, nullArr, nullArr, nullArr,
                                                             nullArr, nullArr)
        
    if k != "":
        (k1, k2, k3, kiso, kauto, fwhmK, starK, sexflagK) = np.loadtxt(k, usecols = columnsX, unpack = True, comments = '#')
        fwhmK = fwhmK * 3600.0
    else:
        (k1, k2, k3, kiso, kauto, fwhmK, starK, sexflagK) = (nullArr, nullArr, nullArr, nullArr, nullArr, nullArr,
                                                             nullArr, nullArr)

    #Convert (x,y) into RA and Dec
    np.savetxt('dummy.foobar', np.c_[(x, y)])
    sp.Popen('xy2sky -d '+rimg+' @dummy.foobar > dummy2.foobar', shell=True).wait()
    (ra, dec) = np.loadtxt('dummy2.foobar', usecols = (0, 1), unpack = True)
    os.remove('dummy.foobar')
    os.remove('dummy2.foobar')
    
    data = {'ra':ra, 'dec':dec, 'x':x, 'y':y, 'b1':b1, 'b2':b2, 'b3':b3, 'biso':biso, 'bauto':bauto, 'starB':starB,
            'sexflagB':sexflagB, 'v1':v1, 'v2':v2, 'v3':v3, 'viso':viso, 'vauto':vauto, 'starV':starV, 'sexflagV':sexflagV,
            'r1':r1, 'r2':r2, 'r3':r3, 'riso':riso, 'rauto':rauto, 'starR':starR, 'sexflagR':sexflagR, 'i1':i1, 'i2':i2,
            'i3':i3, 'iiso':iiso, 'iauto':iauto, 'starI':starI, 'sexflagI':sexflagI, 'z1':z1, 'z2':z2, 'z3':z3, 'ziso':ziso,
            'zauto':zauto, 'starz':starz, 'sexflagz':sexflagz, 'k1':k1, 'k2':k2, 'k3':k3, 'kiso':kiso, 'kauto':kauto,
            'starK':starK, 'sexflagK':sexflagK, 'fwhmB':fwhmB, 'fwhmV':fwhmV, 'fwhmR':fwhmR, 'fwhmI':fwhmI, 'fwhmz':fwhmz,
            'fwhmK':fwhmK, 'kron_a':majorax, 'kron_b':minorax, 'isoarea':isoarea}

    return data
            
#-----------------------------------
def updateBCat(b='',r='',rb='',zpb='',imglist='imglist',rsegmap='',oldcat='',null=-99,xmin=0.,xmax=1e6,
              ymin=0.,ymax=1e6,clname='',pixscale=0.238,errborder=50.,includeLDP=False):

    abconvb = vega2AB['bkpno']
    
    (foo1,foo2)=np.loadtxt(r, usecols = (0,1), unpack=True, comments= '#')
    nsrcs=len(foo1)
    nullArr = np.zeros(nsrcs) + null

    (keys,files)=np.loadtxt(imglist,usecols=(0,1),unpack=True,dtype={'names':('keys','files'), 'formats':('S4','S30')})
    imgs={}
    for x in range(len(keys)):
        imgs[keys[x]]=files[x]

    bbkgsub=imgs['bimg'][:-5]+'_bkgsub.fits'

    hdu = pf.open(oldcat)
    oldData = hdu[1].data

    if includeLDP == True:
        #Add in LDP redshifts
        megatab=pf.open('/Users/tyler/megacat.v5.7.fits')
        megadat=megatab[1].data
        (mzldp,mq)=(megadat['zldp'],megadat['q'])

        wfiSky=coordinates.SkyCoord(ra=oldData['ra']*u.degree, dec=oldData['dec']*u.degree)
        megaSky=coordinates.SkyCoord(ra=megadat['ra']*u.degree, dec=megadat['dec']*u.degree)
        idx, d2d, _ = megaSky.match_to_catalog_sky(wfiSky)
        match = np.where(d2d.arcsec <= 0.5)
        
        (oldData['zLDP'][idx][match],oldData['Q'][idx][match])=(mzldp[match],mq[match])

    (oldRx,oldRy) = (oldData['x'],oldData['y'])

    tmpData = mergeLists(b=b,r=r,zpb=zpb,null=null,rimg=imgs['rimg'])

    res=matchxy(tmpData['x'],tmpData['y'],oldRx,oldRy)

    newData = {'b1':tmpData['b1'][res], 'b2':tmpData['b2'][res], 'b3':tmpData['b3'][res],
               'biso':tmpData['biso'][res], 'bauto':tmpData['bauto'][res],
               'r1':tmpData['r1'][res], 'r2':tmpData['r2'][res], 'r3':tmpData['r3'][res],
               'riso':tmpData['riso'][res], 'rauto':tmpData['rauto'][res],
               'starB':tmpData['starB'][res], 'sexflagB':tmpData['sexflagB'][res],
               'fwhmB':tmpData['fwhmB'][res]}

    #Convert the B-band fluxes from SExtractor into uJy fluxes.
    #Update the dictionary appropriately. Correct
    #the B-band data for seeing if its seeing is worse than the worst
    #WFI seeing image.
    photcols = (4, 5, 6, 7, 8)

    (b1m, b2m, b3m, bisom, bautom) = (calflux(newData['b1'], zpb, abconv = abconvb),
                                      calflux(newData['b2'], zpb, abconv = abconvb),
                                        calflux(newData['b3'], zpb, abconv = abconvb),
                                        calflux(newData['biso'], zpb, abconv = abconvb),
                                        calflux(newData['bauto'], zpb, abconv = abconvb))
    (b1c, b2c, b3c, bisoc, bautoc) = (newData['b1'], newData['b2'], newData['b3'],
                                      newData['biso'], newData['bauto'])

#    pdb.set_trace()

    #Calculate uncertainties
    print '\n\tCalculating uncertainties...\n'

    (auton,ison,n1,n2,n3)=(np.sqrt(np.pi*tmpData['kron_a'][res]*tmpData['kron_b'][res]),
                           np.sqrt(tmpData['isoarea'][res]),np.sqrt(np.pi*(1.0/pixscale)**2.0),
                           np.sqrt(np.pi*(2.0/pixscale)**2.0),np.sqrt(np.pi*(3.0/pixscale)**2.0))
    
    berrpars = lb.main(bbkgsub, rsegmap, outplot='bdepth.pdf', clname=clname, pixscale=pixscale,
                       border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
    bautoerr=(sigfunc(auton,berrpars[0],berrpars[1],berrpars[2])/bautoc)
    bisoerr=(sigfunc(ison,berrpars[0],berrpars[1],berrpars[2])/bisoc)
    b1err=(sigfunc(n1,berrpars[0],berrpars[1],berrpars[2])/b1c)
    b2err=(sigfunc(n2,berrpars[0],berrpars[1],berrpars[2])/b2c)
    b3err=(sigfunc(n3,berrpars[0],berrpars[1],berrpars[2])/b3c)
    print '\tB-band done\n'

    #Correct for seeing (if necessary)
            
    if rb != "":

        (rb1, rb2, rb3, rbiso, rbauto) = np.loadtxt(rb, usecols = photcols, unpack = True, comments = '#')
         
        (r1zp, r2zp, r3zp, risozp, rautozp) = (backZP(newData['r1'],flux2mag(oldData['fR1'],23.9)),
                                               backZP(newData['r2'],flux2mag(oldData['fR2'],23.9)),
                                               backZP(newData['r3'],flux2mag(oldData['fR3'],23.9)),
                                               backZP(newData['riso'],flux2mag(oldData['fRiso'],23.9)),
                                               backZP(newData['rauto'],flux2mag(oldData['fRauto'],23.9)))
         
        (rb1m, rb2m , rb3m, rbisom, rbautom) = (calflux(rb1[res], r1zp), calflux(rb2[res], r2zp),calflux(rb3[res], r3zp),
                                                calflux(rbiso[res], risozp),calflux(rbauto[res], rautozp))

        #Calculate erorrs again! >_<
        rbkgsub=imgs['simg'][:-5]+'_bkgsub.fits'
        rerrpars = lb.main(rbkgsub, rsegmap, outplot='r_smooth_depth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        rautoerr=(sigfunc(auton,rerrpars[0],rerrpars[1],rerrpars[2])/rb1[res])*rbautom
        risoerr=(sigfunc(ison,rerrpars[0],rerrpars[1],rerrpars[2])/rbiso[res])*rbisom
        r1err=(sigfunc(n1,rerrpars[0],rerrpars[1],rerrpars[2])/rb1[res])*rb1m
        r2err=(sigfunc(n2,rerrpars[0],rerrpars[1],rerrpars[2])/rb2[res])*rb2m
        r3err=(sigfunc(n3,rerrpars[0],rerrpars[1],rerrpars[2])/rb3[res])*rb3m
        print '\tSmoothed R-band done\n'

#        pdb.set_trace()
        (b1mc, b2mc, b3mc, bisomc, bautomc) = (seeingCorr(oldData['fR1'], rb1m, b1m), seeingCorr(oldData['fR2'], rb2m, b1m),
                                                seeingCorr(oldData['fR3'], rb3m, b3m), seeingCorr(oldData['fRiso'], rbisom, bisom),
                                                seeingCorr(oldData['fRauto'], rbautom, bautom))

        (b1ecorr,b2ecorr,b3ecorr,bisoecorr,bautoecorr) = (b1err*b1mc, b2err*b2mc, b3err*b3mc,
                                                          bisoerr*bisomc, bautoerr*bautomc)
        
        (newData['b1e'],newData['b2e'],newData['b3e'],
         newData['bisoe'],newData['bautoe']) = (addquad(b1ecorr,oldData['fR1err'],r1err,b1mc,oldData['fR1'],rb1m),
                                                addquad(b2ecorr,oldData['fR2err'],r2err,b2mc,oldData['fR2'],rb2m),
                                                addquad(b3ecorr,oldData['fR3err'],r3err,b3mc,oldData['fR3'],rb3m),
                                                addquad(bisoecorr,oldData['fRisoerr'],risoerr,bisomc,oldData['fRiso'],rbisom),
                                                addquad(bautoecorr,oldData['fRautoerr'],rautoerr,bautomc,oldData['fRauto'],rbautom))

    else:
        (newData['b1e'], newData['b2e'], newData['b3e'],
         newData['bisoe'], newData['bautoe']) = (b1err*newData['b1'], b2err*newData['b2'],
                                                 b3err*newData['b3'], bisoerr*newData['biso'],
                                                 bautoerr*newData['bauto'])
                                                 
        (b1mc, b2mc, b3mc, bisomc, bautomc) = (b1m, b2m, b3m, bisom, bautom)
        
    (newData['b1'], newData['b2'], newData['b3'],
     newData['biso'], newData['bauto']) = (b1mc, b2mc, b3mc, bisomc, bautomc)
        

    outname=clname+'_catalogB_v7.0.fits'

#    final=oldData
#    (final['fwhmB'],final['fB1'],final['fB1err'],final['fB2'],
#     final['fB2err'],final['fB3'],final['fB3err'],final['fBiso'],
#     final['fBisoerr'],final['fBauto'],final['fBautoerr'],
#     final['starB'],final['sexflagB']) = (newData['fwhmB'],newData['b1'],newData['b1e'],newData['b2'],newData['b2e'],
#                                          newData['b3'],newData['b3e'],newData['biso'],newData['bisoe'],
#                                          newData['bauto'],newData['bautoe'],newData['starB'],newData['sexflagB'])

    final={'ids':oldData['ids'],'ra':oldData['ra'],'dec':oldData['dec'],'x':oldData['x'],'y':oldData['y'],'ebv':oldData['ebv'],'fwhmB':oldData['fwhmB'],
           'fwhmV':oldData['fwhmV'],'fwhmR':oldData['fwhmR'],'fwhmI':oldData['fwhmI'],'fwhmz':oldData['fwhmz'],'fwhmK':oldData['fwhmK'],
           'fB1':newData['b1'],'fB1err':newData['b1e'],'fB2':newData['b2'],'fB2err':newData['b2e'],'fB3':newData['b3'],'fB3err':newData['b3e'],
           'fBiso':newData['biso'],'fBisoerr':newData['bisoe'],'fBauto':newData['bauto'],'fBautoerr':newData['bautoe'],'fV1':oldData['fV1'],
           'fV1err':oldData['fV1err'],'fV2':oldData['fV2'],'fV2err':oldData['fV2err'],'fV3':oldData['fV3'],'fV3err':oldData['fV3err'],'fViso':oldData['fViso'],
           'fVisoerr':oldData['fVisoerr'],'fVauto':oldData['fVauto'],'fVautoerr':oldData['fVautoerr'],'fR1':oldData['fR1'],'fR1err':oldData['fR1err'],
           'fR2':oldData['fR2'],'fR2err':oldData['fR2err'],'fR3':oldData['fR3'],'fR3err':oldData['fR3err'],'fRiso':oldData['fRiso'],'fRisoerr':oldData['fRisoerr']
           ,'fRauto':oldData['fRauto'],'fRautoerr':oldData['fRautoerr'],'fI1':oldData['fI1'],'fI1err':oldData['fI1err'],'fI2':oldData['fI2'],
           'fI2err':oldData['fI2err'],'fI3':oldData['fI3'],'fI3err':oldData['fI3err'],'fIiso':oldData['fIiso'],'fIisoerr':oldData['fIisoerr'],
           'fIauto':oldData['fIauto'],'fIautoerr':oldData['fIautoerr'],'fz1':oldData['fz1'],'fz1err':oldData['fz1err'],'fz2':oldData['fz2'],
           'fz2err':oldData['fz2err'],'fz3':oldData['fz3'],'fz3err':oldData['fz3err'],'fziso':oldData['fziso'],'fzisoerr':oldData['fzisoerr'],
           'fzauto':oldData['fzauto'],'fzautoerr':oldData['fzautoerr'],'fK1':oldData['fK1'],'fK1err':oldData['fK1err'],'fK2':oldData['fK2'],
           'fK2err':oldData['fK2err'],'fK3':oldData['fK3'],'fK3err':oldData['fK3err'],'fKiso':oldData['fKiso'],'fKisoerr':oldData['fKisoerr'],
           'fKauto':oldData['fKauto'],'fKautoerr':oldData['fKautoerr'],'zLDP':oldData['zLDP'],'Q':oldData['Q'],'starB':newData['starB'],'starV':oldData['starV'],
           'starR':oldData['starR'],
           'starI':oldData['starI'],'starz':oldData['starz'],'starK':oldData['starK'],'sexflagB':newData['sexflagB'],'sexflagV':oldData['sexflagV'],
           'sexflagR':oldData['sexflagR'],'sexflagI':oldData['sexflagI'],'sexflagz':oldData['sexflagz'],'sexflagK':oldData['sexflagK']}

#    s=['ids','ra','dec','x','y','ebv','fwhmB','fwhmV','fwhmR','fwhmI','fwhmz','fwhmK','fB1','fB1err','fB2',
#                              'fB2err','fB3','fB3err','fBiso','fBisoerr',
#                             'fBauto','fBautoerr','fV1','fV1err','fV2','fV2err','fV3','fV3err','fViso','fVisoerr','fVauto',
#                             'fVautoerr','fR1','fR1err','fR2','fR2err','fR3','fR3err','fRiso','fRisoerr','fRauto',
#                             'fRautoerr','fI1','fI1err','fI2','fI2err','fI3','fI3err','fIiso','fIisoerr','fIauto',
#                             'fIautoerr','fz1','fz1err','fz2','fz2err','fz3','fz3err','fziso','fzisoerr','fzauto',
#                             'fzautoerr','fK1','fK1err','fK2','fK2err','fK3','fK3err','fKiso','fKisoerr','fKauto',
#                             'fKautoerr','zLDP','Q','starB','starV','starR','starI','starz','starK','sexflagB','sexflagV','sexflagR','sexflagI',
#                             'sexflagz','sexflagK']
#    pdb.set_trace()
    #Save the dictionary as a FITS table
    tab = Table(final, names=('ids','ra','dec','x','y','ebv','fwhmB','fwhmV','fwhmR','fwhmI','fwhmz','fwhmK','fB1','fB1err','fB2',
                              'fB2err','fB3','fB3err','fBiso','fBisoerr',
                             'fBauto','fBautoerr','fV1','fV1err','fV2','fV2err','fV3','fV3err','fViso','fVisoerr','fVauto',
                             'fVautoerr','fR1','fR1err','fR2','fR2err','fR3','fR3err','fRiso','fRisoerr','fRauto',
                             'fRautoerr','fI1','fI1err','fI2','fI2err','fI3','fI3err','fIiso','fIisoerr','fIauto',
                             'fIautoerr','fz1','fz1err','fz2','fz2err','fz3','fz3err','fziso','fzisoerr','fzauto',
                             'fzautoerr','fK1','fK1err','fK2','fK2err','fK3','fK3err','fKiso','fKisoerr','fKauto',
                             'fKautoerr','zLDP','Q','starB','starV','starR','starI','starz','starK','sexflagB','sexflagV','sexflagR','sexflagI',
                             'sexflagz','sexflagK'))
    tab.write(outname, format='fits', overwrite=True)

#-----------------------------------
def updateKCat(k='',r='',rk='',zpk='',imglist='imglist',rsegmap='',oldcat='',null=-99,xmin=0.,xmax=1e6,
              ymin=0.,ymax=1e6,clname='',pixscale=0.238,errborder=50.,expmap=''):

    abconvk = vega2AB['k']
    
    (foo1,foo2)=np.loadtxt(r, usecols = (0,1), unpack=True, comments= '#')
    nsrcs=len(foo1)
    nullArr = np.zeros(nsrcs) + null

    (keys,files)=np.loadtxt(imglist,usecols=(0,1),unpack=True,dtype={'names':('keys','files'), 'formats':('S4','S30')})
    imgs={}
    for x in range(len(keys)):
        imgs[keys[x]]=files[x]

    bbkgsub=imgs['kimg'][:-5]+'_bkgsub.fits'

    hdu = pf.open(oldcat)
    oldData = hdu[1].data

    (oldRx,oldRy) = (oldData['x'],oldData['y'])

    tmpData = mergeLists(k=k,r=r,zpk=zpk,null=null,rimg=imgs['rimg'])

    res=matchxy(oldRx,oldRy,tmpData['x'],tmpData['y'])

    newData = {'k1':tmpData['k1'][res], 'k2':tmpData['k2'][res], 'k3':tmpData['k3'][res],
               'kiso':tmpData['kiso'][res], 'kauto':tmpData['kauto'][res],
               'r1':tmpData['r1'][res], 'r2':tmpData['r2'][res], 'r3':tmpData['r3'][res],
               'riso':tmpData['riso'][res], 'rauto':tmpData['rauto'][res],
               'starK':tmpData['starK'][res], 'sexflagK':tmpData['sexflagK'][res],
               'fwhmK':tmpData['fwhmK'][res]}

    #Convert the K-band fluxes from SExtractor into uJy fluxes.
    #Update the dictionary appropriately. Correct
    #the B-band data for seeing if its seeing is worse than the worst
    #WFI seeing image.
    photcols = (4, 5, 6, 7, 8)

    (k1m, k2m, k3m, kisom, kautom) = (calflux(newData['k1'], zpk, abconv = abconvk),
                                      calflux(newData['k2'], zpk, abconv = abconvk),
                                        calflux(newData['k3'], zpk, abconv = abconvk),
                                        calflux(newData['kiso'], zpk, abconv = abconvk),
                                        calflux(newData['kauto'], zpk, abconv = abconvk))
    (k1c, k2c, k3c, kisoc, kautoc) = (newData['k1'], newData['k2'], newData['k3'],
                                      newData['kiso'], newData['kauto'])

    #Calculate uncertainties
    print '\n\tCalculating uncertainties...\n'

    (auton,ison,n1,n2,n3)=(np.sqrt(np.pi*tmpData['kron_a'][res]*tmpData['kron_b'][res]),
                           np.sqrt(tmpData['isoarea'][res]),np.sqrt(np.pi*(1.0/pixscale)**2.0),
                           np.sqrt(np.pi*(2.0/pixscale)**2.0),np.sqrt(np.pi*(3.0/pixscale)**2.0))
    
    kerrpars = lb.main(bbkgsub, rsegmap, outplot='kdepth.pdf', clname=clname, pixscale=pixscale,
                       border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.,
                       expmap=expmap)
    kautoerr=(sigfunc(auton,kerrpars[0],kerrpars[1],kerrpars[2])/kautoc)
    kisoerr=(sigfunc(ison,kerrpars[0],kerrpars[1],kerrpars[2])/kisoc)
    k1err=(sigfunc(n1,kerrpars[0],kerrpars[1],kerrpars[2])/k1c)
    k2err=(sigfunc(n2,kerrpars[0],kerrpars[1],kerrpars[2])/k2c)
    k3err=(sigfunc(n3,kerrpars[0],kerrpars[1],kerrpars[2])/k3c)
    print '\tK-band done\n'

    #Correct for seeing (if necessary)
            
    if rk != "":

        (rk1, rk2, rk3, rkiso, rkauto) = np.loadtxt(rk, usecols = photcols, unpack = True, comments = '#')
         
        (r1zp, r2zp, r3zp, risozp, rautozp) = (backZP(newData['r1'],flux2mag(oldData['fR1'],23.9)),
                                               backZP(newData['r2'],flux2mag(oldData['fR2'],23.9)),
                                               backZP(newData['r3'],flux2mag(oldData['fR3'],23.9)),
                                               backZP(newData['riso'],flux2mag(oldData['fRiso'],23.9)),
                                               backZP(newData['rauto'],flux2mag(oldData['fRauto'],23.9)))
         
        (rk1m, rk2m , rk3m, rkisom, rkautom) = (calflux(rk1[res], r1zp), calflux(rk2[res], r2zp),calflux(rk3[res], r3zp),
                                                calflux(rkiso[res], risozp),calflux(rkauto[res], rautozp))

        #Calculate erorrs again! >_<
        rbkgsub=imgs['simg'][:-5]+'_bkgsub.fits'
        rerrpars = lb.main(rbkgsub, rsegmap, outplot='r_smooth_depth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        rautoerr=(sigfunc(auton,rerrpars[0],rerrpars[1],rerrpars[2])/rk1[res])*rkautom
        risoerr=(sigfunc(ison,rerrpars[0],rerrpars[1],rerrpars[2])/rkiso[res])*rkisom
        r1err=(sigfunc(n1,rerrpars[0],rerrpars[1],rerrpars[2])/rk1[res])*rk1m
        r2err=(sigfunc(n2,rerrpars[0],rerrpars[1],rerrpars[2])/rk2[res])*rk2m
        r3err=(sigfunc(n3,rerrpars[0],rerrpars[1],rerrpars[2])/rk3[res])*rk3m
        print '\tSmoothed R-band done\n'

        (k1mc, k2mc, k3mc, kisomc, kautomc) = (seeingCorr(oldData['fR1'], rk1m, k1m), seeingCorr(oldData['fR2'], rk2m, k1m),
                                                seeingCorr(oldData['fR3'], rk3m, k3m), seeingCorr(oldData['fRiso'], rkisom, kisom),
                                                seeingCorr(oldData['fRauto'], rkautom, kautom))

        (k1ecorr,k2ecorr,k3ecorr,kisoecorr,kautoecorr) = (k1err*k1mc, k2err*k2mc, k3err*k3mc,
                                                          kisoerr*kisomc, kautoerr*kautomc)
        
        (newData['k1e'],newData['k2e'],newData['k3e'],
         newData['kisoe'],newData['kautoe']) = (addquad(k1ecorr,oldData['fR1err'],r1err,k1mc,oldData['fR1'],rk1m),
                                                addquad(k2ecorr,oldData['fR2err'],r2err,k2mc,oldData['fR2'],rk2m),
                                                addquad(k3ecorr,oldData['fR3err'],r3err,k3mc,oldData['fR3'],rk3m),
                                                addquad(kisoecorr,oldData['fRisoerr'],risoerr,kisomc,oldData['fRiso'],rkisom),
                                                addquad(kautoecorr,oldData['fRautoerr'],rautoerr,kautomc,oldData['fRauto'],rkautom))

    else:
        (newData['k1e'], newData['k2e'], newData['k3e'],
         newData['kisoe'], newData['kautoe']) = (k1err*newData['k1'], k2err*newData['k2'],
                                                 k3err*newData['k3'], kisoerr*newData['kiso'],
                                                 kautoerr*newData['kauto'])
                                                 
        (k1mc, k2mc, k3mc, kisomc, kautomc) = (k1m, k2m, k3m, kisom, kautom)
        
    (newData['k1'], newData['k2'], newData['k3'],
     newData['kiso'], newData['kauto']) = (k1mc, k2mc, k3mc, kisomc, kautomc)

    #Scale uncertainties by the exposure map
    expdata=pf.open(expmap)[0].data
    expmax=np.max(expdata)
    expvalues=expdata[tmpData['y'][res].astype(int),tmpData['x'][res].astype(int)]
    scales=1./np.sqrt(expvalues/expmax)

    (newData['k1e'],newData['k2e'],newData['k3e'],
     newData['kisoe'],newData['kautoe']) = (newData['k1e']*scales, newData['k2e']*scales, newData['k3e']*scales,
                                            newData['kisoe']*scales, newData['kautoe']*scales)
        

    outname=clname+'_catalogK_v7.0.fits'

    #Check if old catalog has more sources than K-band (weird problem, but crept up) and remove spurious sources (should only be 1 or 2 max)
    if len(oldData['x']) > len(res):
        use=matchxy(tmpData['x'],tmpData['y'],oldData['x'],oldData['y'])
    else:
        use=np.arange(len(oldData['x']))

    final={'ids':oldData['ids'][use],'ra':oldData['ra'][use],'dec':oldData['dec'][use],'x':oldData['x'][use],'y':oldData['y'][use],'ebv':oldData['ebv'][use],'fwhmB':oldData['fwhmB'][use],
           'fwhmV':oldData['fwhmV'][use],'fwhmR':oldData['fwhmR'][use],'fwhmI':oldData['fwhmI'][use],'fwhmz':oldData['fwhmz'][use],'fwhmK':newData['fwhmK'],
           'fB1':oldData['fB1'][use],'fB1err':oldData['fB1err'][use],'fB2':oldData['fB2'][use],'fB2err':oldData['fB2err'][use],'fB3':oldData['fB3'][use],'fB3err':oldData['fB3err'][use],
           'fBiso':oldData['fBiso'][use],'fBisoerr':oldData['fBisoerr'][use],'fBauto':oldData['fBauto'][use],'fBautoerr':oldData['fBautoerr'][use],'fV1':oldData['fV1'][use],
           'fV1err':oldData['fV1err'][use],'fV2':oldData['fV2'][use],'fV2err':oldData['fV2err'][use],'fV3':oldData['fV3'][use],'fV3err':oldData['fV3err'][use],'fViso':oldData['fViso'][use],
           'fVisoerr':oldData['fVisoerr'][use],'fVauto':oldData['fVauto'][use],'fVautoerr':oldData['fVautoerr'][use],'fR1':oldData['fR1'][use],'fR1err':oldData['fR1err'][use],
           'fR2':oldData['fR2'][use],'fR2err':oldData['fR2err'][use],'fR3':oldData['fR3'][use],'fR3err':oldData['fR3err'][use],'fRiso':oldData['fRiso'][use],'fRisoerr':oldData['fRisoerr'][use]
           ,'fRauto':oldData['fRauto'][use],'fRautoerr':oldData['fRautoerr'][use],'fI1':oldData['fI1'][use],'fI1err':oldData['fI1err'][use],'fI2':oldData['fI2'][use],
           'fI2err':oldData['fI2err'][use],'fI3':oldData['fI3'][use],'fI3err':oldData['fI3err'][use],'fIiso':oldData['fIiso'][use],'fIisoerr':oldData['fIisoerr'][use],
           'fIauto':oldData['fIauto'][use],'fIautoerr':oldData['fIautoerr'][use],'fz1':oldData['fz1'][use],'fz1err':oldData['fz1err'][use],'fz2':oldData['fz2'][use],
           'fz2err':oldData['fz2err'][use],'fz3':oldData['fz3'][use],'fz3err':oldData['fz3err'][use],'fziso':oldData['fziso'][use],'fzisoerr':oldData['fzisoerr'][use],
           'fzauto':oldData['fzauto'][use],'fzautoerr':oldData['fzautoerr'][use],'fK1':newData['k1'],'fK1err':newData['k1e'],'fK2':newData['k2'],
           'fK2err':newData['k2e'],'fK3':newData['k3'],'fK3err':newData['k3e'],'fKiso':newData['kiso'],'fKisoerr':newData['kisoe'],
           'fKauto':newData['kauto'],'fKautoerr':newData['kautoe'],'zLDP':oldData['zLDP'][use],'Q':oldData['Q'][use],'starB':oldData['starB'][use],'starV':oldData['starV'][use],
           'starR':oldData['starR'][use],
           'starI':oldData['starI'][use],'starz':oldData['starz'][use],'starK':newData['starK'],'sexflagB':oldData['sexflagB'][use],'sexflagV':oldData['sexflagV'][use],
           'sexflagR':oldData['sexflagR'][use],'sexflagI':oldData['sexflagI'][use],'sexflagz':oldData['sexflagz'][use],'sexflagK':newData['sexflagK']}

    tab = Table(final, names=('ids','ra','dec','x','y','ebv','fwhmB','fwhmV','fwhmR','fwhmI','fwhmz','fwhmK','fB1','fB1err','fB2',
                              'fB2err','fB3','fB3err','fBiso','fBisoerr',
                             'fBauto','fBautoerr','fV1','fV1err','fV2','fV2err','fV3','fV3err','fViso','fVisoerr','fVauto',
                             'fVautoerr','fR1','fR1err','fR2','fR2err','fR3','fR3err','fRiso','fRisoerr','fRauto',
                             'fRautoerr','fI1','fI1err','fI2','fI2err','fI3','fI3err','fIiso','fIisoerr','fIauto',
                             'fIautoerr','fz1','fz1err','fz2','fz2err','fz3','fz3err','fziso','fzisoerr','fzauto',
                             'fzautoerr','fK1','fK1err','fK2','fK2err','fK3','fK3err','fKiso','fKisoerr','fKauto',
                             'fKautoerr','zLDP','Q','starB','starV','starR','starI','starz','starK','sexflagB','sexflagV','sexflagR','sexflagI',
                             'sexflagz','sexflagK'))
    tab.write(outname, format='fits', overwrite=True)

#-----------------------------------
def main(b="", v="", r="", i="", z="", k="", rb="", rk="", imglist='', rsegmap="",
         zpb=0.0, zpk=0.0, zpz=0.0, null=-99, kpno = False, clname="",pixscale=0.238,
         outprefix='',idname='',synth=False, megacat='/Volumes/BAHAMUT/megacat.v5.7.fits',maxz=100.,
         xmin=-99,xmax=-99,ymin=-99,ymax=-99,errborder=50.0):
    """
    """

    (keys,files)=np.loadtxt(imglist,usecols=(0,1),unpack=True,dtype={'names':('keys','files'), 'formats':('S4','S30')})
    imgs={}
    for x in range(len(keys)):
        imgs[keys[x]]=files[x]

    if 'bimg' in imgs:
        bbkgsub=imgs['bimg'][:-5]+'_bkgsub.fits'
    if 'vimg' in imgs:
        vbkgsub=imgs['vimg'][:-5]+'_bkgsub.fits'
    if 'rimg' in imgs:
        rbkgsub=imgs['rimg'][:-5]+'_bkgsub.fits'
    if 'iimg' in imgs:
        ibkgsub=imgs['iimg'][:-5]+'_bkgsub.fits'
    if 'zimg' in imgs:
        zbkgsub=imgs['zimg'][:-5]+'_bkgsub.fits'
    if 'kimg' in imgs:
        kbkgsub=imgs['kimg'][:-5]+'_bkgsub.fits'
        
        
    if idname == '':
        idname = raw_input('Please enter an ID name for sources: ')
    
    if outprefix == '':
        outname=clname+'_catalog.fits'
    else:
        outname=outprefix+'.fits'

    if kpno == False:
        zpb = zpb + 7.5
    
    if kpno == True:
        abconvb = vega2AB['bkpno']
    else:
        abconvb = vega2AB['bctio']
        
    #Merge the SExtractor photometry from different filters into one dictionary
    print '\n---------\nMERGING CATALOGS\n---------\n'
    
    data=mergeLists(b=b, v=v, r=r, i=i, z=z, k=k, rimg=imgs['rimg'], zpb=zpb, zpk=zpk, null=null)

    (foo1,foo2)=np.loadtxt(r, usecols = (0,1), unpack=True, comments= '#')
    nsrcs=len(foo1)
    nullArr = np.zeros(nsrcs) + null

    #Add in LDP redshifts for John
    megatab=pf.open(megacat)
    megadat=megatab[1].data
    (mzldp,mq)=(megadat['zldp'],megadat['q'])

    wfiSky=coordinates.SkyCoord(ra=data['ra']*u.degree, dec=data['dec']*u.degree)
    megaSky=coordinates.SkyCoord(ra=megadat['ra']*u.degree, dec=megadat['dec']*u.degree)
    idx, d2d, _ = wfiSky.match_to_catalog_sky(megaSky)
    match = np.where(d2d.arcsec <= 1.0)

    nullArr = np.zeros(len(data['ra'])) - 99.0
    (wzldp,wq) = (nullArr, nullArr)
    (wzldp[match],wq[match]) = (mzldp[idx][match],mq[idx][match])

    #Calibrate the WFI data off of the FORS photometry and update the 'data' dictionary
    #with the new fluxes. Add in the AB zeropoints for the WFI data. Includes r = 1, 2, 3"
    #circular apertures, isophotal photometry, and the SExtractor 'auto' apertures.
    print '\n---------\nCALIBRATING WFI PHOTOMETRY\n---------\n'
    
    (v1f, r1f, i1f, v1zp, r1zp, i1zp,rpflag) = zpWFI(data['ra'], data['dec'], data['v1'], data['r1'], data['i1'], data['v3'],
                                              data['r3'], data['i3'], data['starR'],synth=synth)
    print '\t r=1" done\n'
    (v2f, r2f, i2f, v2zp, r2zp, i2zp,_) = zpWFI(data['ra'], data['dec'], data['v2'], data['r2'], data['i2'], data['v3'],
                                              data['r3'], data['i3'], data['starR'],synth=synth)
    print '\t r=2" done\n'
    (v3f, r3f, i3f, v3zp, r3zp, i3zp,_) = zpWFI(data['ra'], data['dec'], data['v3'], data['r3'], data['i3'], data['v3'],
                                              data['r3'], data['i3'], data['starR'],synth=synth)
    print '\t r=3" done\n'
    (visof, risof, iisof, visozp, risozp, iisozp,_) = zpWFI(data['ra'], data['dec'], data['viso'], data['riso'], data['iiso'],
                                                          data['v3'], data['r3'], data['i3'], data['starR'],synth=synth)
    print '\t r=iso done\n'
    (vautof, rautof, iautof, vautozp, rautozp, iautozp,_) = zpWFI(data['ra'], data['dec'], data['vauto'], data['rauto'],
                                                                data['iauto'], data['v3'], data['r3'], data['i3'],
                                                                data['starR'],synth=synth)
    print '\t r=auto done\n'

    #Save the counts to pass on to the data dictionary
    (v1c, v2c, v3c, visoc, vautoc, r1c, r2c, r3c, risoc, rautoc, i1c, i2c, i3c, iisoc, iautoc) = (data['v1'], data['v2'], data['v3'], data['viso'], data['vauto'],
                                                                                                  data['r1'], data['r2'], data['r3'], data['riso'], data['rauto'],
                                                                                                  data['i1'], data['i2'], data['i3'], data['iiso'], data['iauto'])
        
    (data['v1'], data['v1c'], data['v2'], data['v2c'], data['v3'], data['v3c'], data['viso'], data['visoc'],
    data['vauto'], data['vautoc'], data['r1'], data['r1c'], data['r2'], data['r2c'], data['r3'], data['r3c'],
    data['riso'], data['risoc'], data['rauto'], data['rautoc'], data['i1'], data['i1c'], data['i2'], data['i2c'],
    data['i3'], data['i3c'], data['iiso'], data['iisoc'], data['iauto'], data['iautoc']) = (v1f, v1c, v2f, v2c, v3f, v3c, visof, visoc,
                                                                                            vautof, vautoc, r1f, r1c, r2f, r2c, r3f, r3c,
                                                                                            risof, risoc, rautof, rautoc, i1f, i1c, i2f,
                                                                                            i2c, i3f, i3c, iisof, iisoc, iautof, iautoc)

    (data['v1zp'], data['r1zp'], data['i1zp'], data['v2zp'],
     data['r2zp'], data['i2zp'], data['v3zp'], data['r3zp'],
     data['i3zp'], data['visozp'], data['risozp'], data['iisozp'],
     data['vautozp'], data['rautozp'], data['iautozp'], data['rpflag']) = (v1zp, r1zp, i1zp, v2zp, r2zp, i2zp, v3zp, r3zp, i3zp, visozp,
                                                          risozp, iisozp, vautozp, rautozp, iautozp,rpflag)

    #Convert the B- and K-band fluxes from SExtractor into uJy fluxes.
    #Update the 'data' dictionary appropriately. Correct
    #the B- and K-band data for seeing if their seeing is worse than the worst
    #WFI seeing image.
    photcols = (4, 5, 6, 7, 8)
    if b != "":
        (b1m, b2m, b3m, bisom, bautom) = (calflux(data['b1'], zpb, abconv = abconvb), calflux(data['b2'], zpb, abconv = abconvb),
                                          calflux(data['b3'], zpb, abconv = abconvb), calflux(data['biso'], zpb, abconv = abconvb),
                                          calflux(data['bauto'], zpb, abconv = abconvb))
        (b1c, b2c, b3c, bisoc, bautoc) = (data['b1'], data['b2'], data['b3'], data['biso'], data['bauto'])
         
        if rb != "":
            (Rb1, Rb2, Rb3, Rbiso, Rbauto) = np.loadtxt(rb, usecols = photcols, unpack = True, comments = '#')
            (rb1m, rb2m , rb3m, rbisom, rbautom) = (calflux(rb1, data['r1zp'], vega2AB['r']), calflux(rb2, data['r2zp'], vega2AB['r']),
                                                    calflux(rb3, data['r3zp'], vega2AB['r']), calflux(rbiso, data['r2zp'], vega2AB['r']),
                                                    calflux(rbauto, data['rauto'], vega2AB['r']))
            (b1mc, b2mc, b3mc, bisomc, bautomc) = (seeingCorr(data['r1'], rb1m, b1m), seeingCorr(data['r2'], rb2m, b1m),
                                                    seeingCorr(data['r3'], rb3m, b3m), seeingCorr(data['riso'], rbisom, bisom),
                                                    seeingCorr(data['rauto'], rbautom, bautom))
            (data['b1e'],data['b2e'],data['b3e'],data['bisoe'],data['bautoe']) = (addquad(data['k1e'],data['r1e'],b1mc,data['r1']),
                                                                                  addquad(data['k2e'],data['r2e'],b2mc,data['r2']),
                                                                                  addquad(data['k3e'],data['r3e'],b3mc,data['r3']),
                                                                                  addquad(data['kisoe'],data['risoe'],bisomc,data['riso']),
                                                                                  addquad(data['kautoe'],data['rautoe'],bautomc,
                                                                                          data['rauto']))

        else:
            (b1mc, b2mc, b3mc, bisomc, bautomc) = (b1m, b2m, b3m, bisom, bautom)
        (data['b1'], data['b2'], data['b3'], data['biso'], data['bauto']) = (b1mc, b2mc, b3mc, bisomc, bautomc)
        
    if k != "":
        (k1m, k2m, k3m, kisom, kautom) = (calflux(data['k1'], zpk, abconv = vega2AB['k']), calflux(data['k2'], zpk, abconv = vega2AB['k']),
                                          calflux(data['k3'], zpk, abconv = vega2AB['k']), calflux(data['kiso'], zpk, abconv = vega2AB['k']),
                                          calflux(data['kauto'], zpk, abconv = vega2AB['k']))
        (k1c, k2c, k3c, kisoc, kautoc) = (data['k1'], data['k2'], data['k3'], data['kiso'], data['kauto'])

        #Correct for seeing if necessary
        if rk != '':
            (rk1, rk2, rk3, rkiso, rkauto) = np.loadtxt(rk, usecols = photcols, unpack = True, comments = '#')
            (rk1m, rk2m , rk3m, rkisom, rkautom) = (calflux(rk1, data['r1zp'], vega2AB['r']), calflux(rk2, data['r2zp'], vega2AB['r']),
                                                    calflux(rk3, data['r3zp'], vega2AB['r']), calflux(rkiso, data['r2zp'], vega2AB['r']),
                                                    calflux(rkauto, data['rauto'], vega2AB['r']))
            (k1mc, k2mc, k3mc, kisomc, kautomc) = (seeingCorr(data['r1'], rk1m, k1m), seeingCorr(data['r2'], rk2m, k1m),
                                                   seeingCorr(data['r3'], rk3m, k3m), seeingCorr(data['riso'], rkisom, kisom),
                                                   seeingCorr(data['rauto'], rkautom, kautom))
            (data['k1e'],data['k2e'],data['k3e'],data['kisoe'],data['kautoe']) = (addquad(data['k1e'],data['r1e'],k1mc,data['r1']),
                                                                                  addquad(data['k2e'],data['r2e'],k2mc,data['r2']),
                                                                                  addquad(data['k3e'],data['r3e'],k3mc,data['r3']),
                                                                                  addquad(data['kisoe'],data['risoe'],kisomc,data['riso']),
                                                                                  addquad(data['kautoe'],data['rautoe'],kauotmc,
                                                                                          data['rauto']))
        else:
            (k1mc, k2mc, k3mc, kisomc, kautomc) = (k1m, k2m, k3m, kisom, kautom)
        (k1f, k2f, k3f, kisof, kautof) = (ab2ujy(k1mc), ab2ujy(k2mc), ab2ujy(k3mc), ab2ujy(kisomc), ab2ujy(kautomc))
        (data['k1'], data['k2'], data['k3'], data['kiso'], data['kauto']) = (k1f, k2f, k3f, kisof, kautof)

    #Convert z-band from SExtractor to uJy fluxes
    #Note that the z-band observations were calibrated using
    #Smith standards in AB magnitudes, so no AB conversion
    #is necessary
    if z != "":
        (z1c, z2c, z3c, zisoc, zautoc) = (data['z1'], data['z2'], data['z3'], data['ziso'], data['zauto'])
        (data['z1'], data['z2'], data['z3'],
         data['ziso'], data['zauto']) = (calflux(data['z1'], zpz, abconv = 0.0), calflux(data['z2'], zpz, abconv = 0.0),
                                          calflux(data['z3'], zpz, abconv = 0.0), calflux(data['ziso'], zpz, abconv = 0.0),
                                          calflux(data['zauto'], zpz, abconv = 0.0))

#        find=np.where(data['x'] == 8415.469)
#        print data['z1'][find], z1c[find]
#        pdb.set_trace()
        
         
    #Compute errors
    print '\n---------\nCOMPUTING ERRORS\n---------\n'
    (auton,ison,n1,n2,n3)=(np.sqrt(np.pi*data['kron_a']*data['kron_b']),np.sqrt(data['isoarea']),
                           np.sqrt(np.pi*(1.0/pixscale)**2.0),np.sqrt(np.pi*(2.0/pixscale)**2.0),
                           np.sqrt(np.pi*(3.0/pixscale)**2.0))

    zpbarr=np.zeros(len(data['b1']))+zpb
    zpkarr=np.zeros(len(data['k1']))+zpk
    zpzarr=np.zeros(len(data['z1']))+zpz
    
    if 'bimg' in imgs:
        berrpars = lb.main(bbkgsub, rsegmap, outplot='bdepth.pdf', clname=clname, pixscale=pixscale,
                           border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        bautoerr=(sigfunc(auton,berrpars[0],berrpars[1],berrpars[2])/bautoc)*data['bauto']
        bisoerr=(sigfunc(ison,berrpars[0],berrpars[1],berrpars[2])/bisoc)*data['biso']
        b1err=(sigfunc(n1,berrpars[0],berrpars[1],berrpars[2])/b1c)*data['b1']
        b2err=(sigfunc(n2,berrpars[0],berrpars[1],berrpars[2])/b2c)*data['b2']
        b3err=(sigfunc(n3,berrpars[0],berrpars[1],berrpars[2])/b3c)*data['b3']
        print '\t B-band done\n'
    else:
        (berrpars,bautoerr,bisoerr,b1err,b2err,b3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)
     
    if 'vimg' in imgs:
        verrpars = lb.main(vbkgsub, rsegmap, outplot='vdepth.pdf', clname=clname, pixscale=pixscale,
                           border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        vautoerr=(sigfunc(auton,verrpars[0],verrpars[1],verrpars[2])/data['vautoc'])*data['vauto']
        visoerr=(sigfunc(ison,verrpars[0],verrpars[1],verrpars[2])/data['visoc'])*data['viso']
        v1err=(sigfunc(n1,verrpars[0],verrpars[1],verrpars[2])/data['v1c'])*data['v1']
        v2err=(sigfunc(n2,verrpars[0],verrpars[1],verrpars[2])/data['v2c'])*data['v2']
        v3err=(sigfunc(n3,verrpars[0],verrpars[1],verrpars[2])/data['v3c'])*data['v3']
        print '\t V-band done\n'
    else:
        (verrpars,vautoerr,visoerr,v1err,v2err,v3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)
        
    if 'rimg' in imgs:    
        rerrpars = lb.main(rbkgsub, rsegmap, outplot='rdepth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        rautoerr=(sigfunc(auton,rerrpars[0],rerrpars[1],rerrpars[2])/data['rautoc'])*data['rauto']
        risoerr=(sigfunc(ison,rerrpars[0],rerrpars[1],rerrpars[2])/data['risoc'])*data['riso']
        r1err=(sigfunc(n1,rerrpars[0],rerrpars[1],rerrpars[2])/data['r1c'])*data['r1']
        r2err=(sigfunc(n2,rerrpars[0],rerrpars[1],rerrpars[2])/data['r2c'])*data['r2']
        r3err=(sigfunc(n3,rerrpars[0],rerrpars[1],rerrpars[2])/data['r3c'])*data['r3']
        print '\t R-band done\n'
    else:
        (rerrpars,rautoerr,risoerr,r1err,r2err,r3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)
        
    if 'iimg' in imgs:
        ierrpars = lb.main(ibkgsub, rsegmap, outplot='idepth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        iautoerr=(sigfunc(auton,ierrpars[0],ierrpars[1],ierrpars[2])/data['iautoc'])*data['iauto']
        iisoerr=(sigfunc(ison,ierrpars[0],ierrpars[1],ierrpars[2])/data['iisoc'])*data['iiso']
        i1err=(sigfunc(n1,ierrpars[0],ierrpars[1],ierrpars[2])/data['i1c'])*data['i1']
        i2err=(sigfunc(n2,ierrpars[0],ierrpars[1],ierrpars[2])/data['i2c'])*data['i2']
        i3err=(sigfunc(n3,ierrpars[0],ierrpars[1],ierrpars[2])/data['i3c'])*data['i3']
        print '\t I-band done\n'
    else:
        (ierrpars,iautoerr,iisoerr,i1err,i2err,i3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)

    if 'zimg' in imgs:
        zerrpars = lb.main(zbkgsub, rsegmap, outplot='zdepth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=False, aprange=[0.5,2.0],maxrange=500.)
        zautoerr=(sigfunc(auton,zerrpars[0],zerrpars[1],zerrpars[2])/zautoc)*data['zauto']
        zisoerr=(sigfunc(ison,zerrpars[0],zerrpars[1],zerrpars[2])/zisoc)*data['ziso']
        z1err=(sigfunc(n1,zerrpars[0],zerrpars[1],zerrpars[2])/z1c)*data['z1']
        z2err=(sigfunc(n2,zerrpars[0],zerrpars[1],zerrpars[2])/z2c)*data['z2']
        z3err=(sigfunc(n3,zerrpars[0],zerrpars[1],zerrpars[2])/z3c)*data['z3']
        print '\t z-band done\n'
    else:
        (zerrpars,zautoerr,zisoerr,z1err,z2err,z3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)
        
    if 'kimg' in imgs:
        kerrpars = lb.main(kbkgsub, rsegmap, outplot='kdepth.pdf', clname=clname, pixscale=pixscale,
                            border=errborder, persec=True, aprange=[0.5,2.0])
        kautoerr=(sigfunc(auton,kerrpars[0],kerrpars[1],kerrpars[2])/kautoc)*data['kauto']
        kisoerr=(sigfunc(ison,kerrpars[0],kerrpars[1],kerrpars[2])/kisoc)*data['kiso']
        k1err=(sigfunc(n1,kerrpars[0],kerrpars[1],kerrpars[2])/k1c)*data['k1']
        k2err=(sigfunc(n2,kerrpars[0],kerrpars[1],kerrpars[2])/k2c)*data['k2']
        k3err=(sigfunc(n3,kerrpars[0],kerrpars[1],kerrpars[2])/k3c)*data['k3']
        print '\t K-band done\n'
    else:
        (kerrpars,kautoerr,kisoerr,k1err,k2err,k3err)=(nullArr,nullArr,nullArr,nullArr,nullArr,nullArr)
    
    (data['bautoerr'],data['bisoerr'],data['b1err'],data['b2err'],data['b3err'],data['vautoerr'],data['visoerr'],
     data['v1err'],data['v2err'],data['v3err'],data['rautoerr'],data['risoerr'],data['r1err'],data['r2err'],
     data['r3err'],data['iautoerr'],data['iisoerr'],data['i1err'],data['i2err'],data['i3err'],data['z1err'],
     data['z2err'],data['z3err'],data['zisoerr'],data['zautoerr'],data['kautoerr'],
     data['kisoerr'],data['k1err'],data['k2err'],data['k3err']) = (bautoerr,bisoerr,b1err,b2err,b3err,vautoerr,visoerr,
                                                                   v1err,v2err,v3err,rautoerr,risoerr,r1err,r2err,
                                                                   r3err,iautoerr,iisoerr,i1err,i2err,i3err,z1err,
                                                                   z2err,z3err,zisoerr,zautoerr,kautoerr,
                                                                   kisoerr,k1err,k2err,k3err)

    #Look up the E(B-V) values from the Schlegel dust maps
    data['ebv']=getEBV(data['ra'],data['dec'])
    
    #Update the dictionary with flags
    #'b1','b2','b3','biso','bauto',
    fs=['v1','v2','v3','viso','vauto','r1','r2','r3',
        'riso','rauto','i1','i2','i3','iiso','iauto','z1','z2','z3','ziso','zauto',
        'k1','k2','k3','kiso','kauto']

    #'b1err','b2err','b3err','bisoerr','bautoerr',
    es=['v1err','v2err','v3err',
        'visoerr','vautoerr','r1err','r2err','r3err','risoerr','rautoerr','i1err',
        'i2err','i3err','iisoerr','iautoerr','z1err','z2err','z3err','zisoerr',
        'zautoerr','k1err','k2err','k3err','kisoerr','kautoerr']

    if xmin == -99:
        xmin=np.min(data['x'])
    if xmax == -99:
        xmax=np.max(data['x'])
    if ymin == -99:
        ymin=np.min(data['y'])
    if ymax == -99:
        ymax=np.max(data['y'])
        
    for xx in range(len(fs)):
        (data[fs[xx]],data[es[xx]])=flags(data[fs[xx]],data[es[xx]],data['x'],data['y'],xmin=xmin,xmax=xmax,ymin=ymin,
                                          ymax=ymax)
    
    data = fixData(data)

    good = np.where(data['r1'] != -55)
    
    ids=[]
    for x in range(len(good[0])):
        ids.append('WFI_'+idname+'_'+str(int(x+1)))

    final={'ids':ids,'ra':data['ra'][good],'dec':data['dec'][good],'x':data['x'][good],'y':data['y'][good],'ebv':data['ebv'][good],'fwhmB':data['fwhmB'][good],
           'fwhmV':data['fwhmV'][good],'fwhmR':data['fwhmR'][good],'fwhmI':data['fwhmI'][good],'fwhmz':data['fwhmz'][good],'fwhmK':data['fwhmK'][good],
           'fB1':data['b1'][good],'fB1err':data['b1err'][good],'fB2':data['b2'][good],'fB2err':data['b2err'][good],'fB3':data['b3'][good],'fB3err':data['b3err'][good],
           'fBiso':data['biso'][good],'fBisoerr':data['bisoerr'][good],'fBauto':data['bauto'][good],'fBautoerr':data['bautoerr'][good],'fV1':data['v1'][good],
           'fV1err':data['v1err'][good],'fV2':data['v2'][good],'fV2err':data['v2err'][good],'fV3':data['v3'][good],'fV3err':data['v3err'][good],'fViso':data['viso'][good],
           'fVisoerr':data['visoerr'][good],'fVauto':data['vauto'][good],'fVautoerr':data['vautoerr'][good],'fR1':data['r1'][good],'fR1err':data['r1err'][good],
           'fR2':data['r2'][good],'fR2err':data['r2err'][good],'fR3':data['r3'][good],'fR3err':data['r3err'][good],'fRiso':data['riso'][good],'fRisoerr':data['risoerr'][good]
           ,'fRauto':data['rauto'][good],'fRautoerr':data['rautoerr'][good],'fI1':data['i1'][good],'fI1err':data['i1err'][good],'fI2':data['i2'][good],
           'fI2err':data['i2err'][good],'fI3':data['i3'][good],'fI3err':data['i3err'][good],'fIiso':data['iiso'][good],'fIisoerr':data['iisoerr'][good],
           'fIauto':data['iauto'][good],'fIautoerr':data['iautoerr'][good],'fz1':data['z1'][good],'fz1err':data['z1err'][good],'fz2':data['z2'][good],
           'fz2err':data['z2err'][good],'fz3':data['z3'][good],'fz3err':data['z3err'][good],'fziso':data['ziso'][good],'fzisoerr':data['zisoerr'][good],
           'fzauto':data['zauto'][good],'fzautoerr':data['zautoerr'][good],'fK1':data['k1'][good],'fK1err':data['k1err'][good],'fK2':data['k2'][good],
           'fK2err':data['k2err'][good],'fK3':data['k3'][good],'fK3err':data['k3err'][good],'fKiso':data['kiso'][good],'fKisoerr':data['kisoerr'][good],
           'fKauto':data['kauto'][good],'fKautoerr':data['kautoerr'][good],'zLDP':wzldp[good],'Q':wq[good],'starB':data['starB'][good],'starV':data['starV'][good],'starR':data['starR'][good],
           'starI':data['starI'][good],'starz':data['starz'][good],'starK':data['starK'][good],'sexflagB':data['sexflagB'][good],'sexflagV':data['sexflagV'][good],
           'sexflagR':data['sexflagR'][good],'sexflagI':data['sexflagI'][good],'sexflagz':data['sexflagz'][good],'sexflagK':data['sexflagK'][good],
           'r1counts':data['r1c'][good],'rpflag':data['rpflag'][good]}
    
    #Save the dictionary as a FITS table
    tab = Table(final, names=('ids','ra','dec','x','y','ebv','fwhmB','fwhmV','fwhmR','fwhmI','fwhmz','fwhmK','fB1','fB1err','fB2',
                              'fB2err','fB3','fB3err','fBiso','fBisoerr',
                             'fBauto','fBautoerr','fV1','fV1err','fV2','fV2err','fV3','fV3err','fViso','fVisoerr','fVauto',
                             'fVautoerr','fR1','fR1err','fR2','fR2err','fR3','fR3err','fRiso','fRisoerr','fRauto',
                             'fRautoerr','fI1','fI1err','fI2','fI2err','fI3','fI3err','fIiso','fIisoerr','fIauto',
                             'fIautoerr','fz1','fz1err','fz2','fz2err','fz3','fz3err','fziso','fzisoerr','fzauto',
                             'fzautoerr','fK1','fK1err','fK2','fK2err','fK3','fK3err','fKiso','fKisoerr','fKauto',
                             'fKautoerr','zLDP','Q','starB','starV','starR','starI','starz','starK','sexflagB','sexflagV','sexflagR','sexflagI',
                             'sexflagz','sexflagK','r1counts','rpflag'))
    tab.write(outname, format='fits', overwrite=True)


