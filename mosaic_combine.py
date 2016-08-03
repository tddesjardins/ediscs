from pyraf import iraf
from iraf import noao, mscred, stsdas, imgtools, toolbox
import numpy as np, pyfits as pf, shutil, glob, os, subprocess, pdb, shlex
from astLib import astStats
from scipy.stats.mstats import mode
from uncertainties import ufloat,unumpy
from astropy import units as u
from astropy.coordinates import SkyCoord

#--------------------------------------------------------------------------------------------------------
def log(string,logfile):

    print string
    f=open(logfile,'a')
    f.write(string)
    f.close()

#--------------------------------------------------------------------------------------------------------
def file_check(file,delete=False):
    
    if os.path.exists(file):
        res=True
    else:
        res=False

    if res==True:
        if delete==True:
            os.remove(file)

    return res

#--------------------------------------------------------------------------------------------------------
def applywcs(input,cat='ub1',logfile='mosaic_log'):

    print '\n\tAligning '+input+' WCS using '+cat.upper()+' catalog...'

    output=input[:-5]+'_wcs.fits'
    file_check(output,delete=True)

    cmd=shlex.split('/usr/bin/wcstools-3.8.6/bin//imwcs -c '+cat+' -q irs -h 100000 -m 16 24 '+\
            '-i -3 -o '+output+' -wv '+input)
    subprocess.Popen(cmd).wait()

    return output

#--------------------------------------------------------------------------------------------------------
def classifyImages(prepend=''):

    files=glob.glob(prepend+'tu*.fits')

    for x in files:
        hdr=pf.open(x)[0].header
        res=hdr['PRODTYPE']
        if 'image' in res:
            img=x.split('/')[-1]
        elif 'dqmask' in res:
            mask=x.split('/')[-1]

    return (img,mask)

#--------------------------------------------------------------------------------------------------------
def calczp(airmass):#,zpfile='/Users/tyler/research/cluster_photoz/mosaic/zeropoints.dat'):

    (zp0,aircorr)=(26.9162394,-0.3247822)
    return zp0+(aircorr*airmass)

#--------------------------------------------------------------------------------------------------------
def scaleAndCombineMEF(input,mask):

    print input

    hdu=pf.open(input)
    hdr=hdu[0].header
    airmass=hdr['AIRMASS']

    zp=calczp(airmass)
    print '\n\tCalculated Zeropoint: '+str(zp).strip()+' mag'

    copy=input[:-5]+'_copy.fits'
    shutil.copyfile(input,copy)

    hdu=pf.open(copy,mode='update')
    (hdu[0].header['ZEROPT'],hdu[0].header['BPM'])=(zp,mask)
    hdu.flush()
    hdu.close()
        
    iraf.unlearn('mscimage')
    iraf.mscimage.blank=-99.0
    cmd=(copy,copy[:-9]+'singleFrame.fits')
    iraf.mscimage(cmd[0],cmd[1])

    hdu=pf.open(copy[:-9]+'singleFrame.fits',mode='update')
    hdu[0].header['BPM']=copy[:-9]+'singleFrame_bpm.pl'
    hdu.flush()
    hdu.close()

    return copy[:-9]+'singleFrame.fits'

#--------------------------------------------------------------------------------------------------------
def scaleZP(inputList,reference,logfile='mosaic_log'):

    refZP=pf.open(reference)[0].header['ZEROPT']

    print '\nNormalizing all images in stack to common zeropoint...'

    newStack=[]
    for x in inputList:
        print '\n\tWorking on image '+x+'\n'
        copy=x[:-5]+'_tmp.fits'
        hdu=pf.open(x,mode='update')
        imgZP=hdu[0].header['ZEROPT']
        hdu[0].header['ZEROPT']=refZP
        hdu.flush()
        hdu.close()

        if 'n1' in x:
            print '\n\tImage is from night 1, scaling zeropoint to match reference image...\n'        
            scaleFactor=10.0**(-0.4*(imgZP-refZP))
        else:
            print '\n\tImage is from night 2, no zeropoint scaling will be applied...\n'
            scaleFactor=1
        iraf.unlearn('imcalc')
        iraf.imcalc(x,copy,'im1*'+str(scaleFactor),verbose='yes')

        newStack.append(copy)

    return (newStack,refZP)

#--------------------------------------------------------------------------------------------------------
def parseCMD(file):

    l=open(file,'r').readlines()
    d={}
    for x in l:
        data=x.rstrip().split(':')
        d[data[0].strip()]=data[1].strip()

    return d

#--------------------------------------------------------------------------------------------------------
def singleFrame(directory,wcs=True):


    listFiles=['maskList.in','stackList.in']
    (stackFile,stackList,wcsList)=(open(listFiles[1],'a'),[],[])
     
    (image,mask)=classifyImages(prepend=directory+'/')
    print '\nWorking on image in '+directory.strip()

    scaleImage=scaleAndCombineMEF(directory+'/'+image,directory+'/'+mask)

    if wcs == True:
        wcsImage=applywcs(scaleImage)
        stackFile.write(wcsImage+'\n')

        stackFile.close()

#--------------------------------------------------------------------------------------------------------
def scaleToSingleZP(stackfile='stackList.in'):

    wcsList=[]
    for x in open(stackfile,'r').readlines():
        wcsList.append(x.rstrip())
    (stackList,finZP)=scaleZP(wcsList,wcsList[0])

    if os.path.exists('finalList.in'):
        os.remove('finalList.in')
    finList=open('finalList.in','w')
     
    for x in stackList:
        finList.write(x+'[0]\n')
    finList.close()

#--------------------------------------------------------------------------------------------------------
def finalCombine(stackname):

    root=stackname[:-5]
    iraf.unlearn('mscstack')
    iraf.mscstack.bpmasks=root+'_bpm'
    iraf.mscstack.scale='!MSCSCALE'
    iraf.mscstack.zero='!MSCZERO'
    iraf.mscstack.expmasks=root+'_expmap'
    iraf.mscstack.nrejmasks=root+'_nrej'
    iraf.mscstack.sigmas=root+'_sigma'
    iraf.mscstack.combine='average'
    iraf.mscstack.mclip='yes'
    iraf.mscstack.reject='crrej'
    iraf.mscstack.blank=-99.0
    iraf.mscstack.lthreshold='INDEF'
    iraf.mscstack.hthreshold='INDEF'
    iraf.mscstack.hsigma=4.0
    iraf.mscstack.lsigma='INDEF'
    iraf.mscstack.nkeep=3
    iraf.mscstack.grow=2.5
    iraf.mscstack.rdnoise=0.5
    iraf.mscstack.gain=1.0
    iraf.mscstack.masktype='goodvalue'

    files=glob.glob('*'+stackname+'*.fits')
    files2=glob.glob('*'+stackname+'*.pl')
    for x in files:
        if os.path.exists(x):
            os.remove(x)
    for x in files2:
        if os.path.exists(x):
            os.remove(x)

    f=open('finalList.in','r').readlines()
    out=open('finalList2.in','w')
    for x in f:
        out.write(x.rstrip()[:-3]+'\n')
    out.close()
    
    iraf.mscstack('@finalList2.in',stackname)

    os.remove('logfile')
        
    print '\nFinal Stacked Image: '+stackname.strip()
    print '\nBE SURE TO CHECK THE FINAL STACKED IMAGE!!!'

#--------------------------------------------------------------------------------------------------------
def bootstrapZP(image,cat='zp.cat',refcat='ref.cat',refzp=1.0):

    (ra,dec,mag,star)=np.loadtxt(cat,usecols=(0,1,2,3),unpack=True,
                                     dtype=np.dtype([('ra','<f10'),('dec','<f10'),('mag','<f10'),
                                                     ('star','<f10')]))

    (rra,rdec,rmag,rstar)=np.loadtxt(refcat,usecols=(0,1,2,3),unpack=True,
                                            dtype=np.dtype([('rra','<f10'),('rdec','<f10'),('rmag','<f10'),
                                                            ('rstar','<f10')]))

    #Grab only stars from the reference catalog
    refgood=np.where((rstar >= 0.98) & (rmag != 99.0) & (rmag > 17.0) & (rmag < 22.5))
    refcat=SkyCoord(ra=rra[refgood]*u.degree,dec=rdec[refgood]*u.degree)

    #Sort through and remove anything that is not a star and is not isolated
    #from other sources in the input catalog
    catgood=np.where((star >= 0.98) & (mag != 99.0))
    cat=SkyCoord(ra=ra[catgood]*u.degree,dec=dec[catgood]*u.degree)
    idx,d2d,_=cat.match_to_catalog_sky(refcat)
    _,d2d2,_=cat.match_to_catalog_sky(cat,2)
    final=np.where((d2d.arcsec < 0.5) & (d2d2.arcsec >= 5.0))
    
    diff=rmag[refgood][idx][final]-mag[catgood][final]
    imgZP=np.mean(diff)

    print '\n\tUsing '+str(len(diff))+' stars to calculate ZP...'
    print '\tMean ZP: '+str(round(imgZP,3))+' mag\n'

    scaleFactor=10.0**(0.4*(refzp-imgZP))

    iraf.unlearn('imcalc')
    iraf.imcalc(image,image[:-5]+'_scaled.fits','im1*'+str(scaleFactor))

#--------------------------------------------------------------------------------------------------------
def match(ra,dec,mag,ref,tolerance=0.1):

    cdt=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
    refcdt=SkyCoord(ra=ref[0]*u.degree,dec=ref[1]*u.degree)
    idx,sep,_=cdt.match_to_catalog_sky(refcdt,1)
    good=np.where(sep.arcsec < tolerance)

    return(mag[good],ref[2][idx[good]])
    
#--------------------------------------------------------------------------------------------------------
def checkDist(x,y,tolerance=15.0):

    flag=np.ones(len(x))
    cdt=SkyCoord(ra=x*u.degree,dec=y*u.degree)
    ref=SkyCoord(ra=x*u.degree,dec=y*u.degree)
    idx,sep,_=cdt.match_to_catalog_sky(ref,2)
    flag[np.where(sep.arcsec < tolerance)]=0
    
    return flag

#--------------------------------------------------------------------------------------------------------    
def separation(ra1,dec1,ra2,dec2):

    cdt1=SkyCoord(ra=ra1*u.degree,dec=dec1*u.degree)
    cdt2=SkyCoord(ra=ra2*u.degree,dec=dec2*u.degree)

    return cdt1.separation(cdt2).arcsec
    
#--------------------------------------------------------------------------------------------------------
def mkReg(catfile,regfile):

    if os.path.exists(regfile):
        os.remove(regfile)

    out=open(regfile,'w')
    (ra,dec,stellar)=np.loadtxt(catfile,usecols=(0,1,4),unpack=True)
    out.write('fk5\n')
    for x in range(len(ra)):
        if stellar[x] <0.05:
            out.write('circle('+str(ra[x])+','+str(dec[x])+',10") # color=red width=2\n')
    out.close()

#--------------------------------------------------------------------------------------------------------
def mkweightmap(bpm,outweight='weight.fits'):

    iraf.unlearn('imexpr')
    iraf.imexpr('a = 0 ? 1 : 0',outweight,a=bpm)

#--------------------------------------------------------------------------------------------------------
def night2ZP(image,cat,ref):

    zpimg=image[:-5]+'_zp.fits'
    if os.path.exists(zpimg):
        os.remove(zpimg)
    shutil.copy(image,zpimg)
    
    zp=bootstrapZP(cat,refcat=ref,errthresh=20.0)

    hdu=pf.open(zpimg,mode='update')
    hdu[0].header['ZEROPT']=zp.nominal_value
    hdu[0].header['ZEROERR']=zp.std_dev
    hdu.flush()
    hdu.close()

    print 'Updated '+zpimg+' with zeropoint = '+str(zp)

#--------------------------------------------------------------------------------------------------------
def copylist(file,outfile='stackList.in'):

    out=open(outfile,'w')
    for x in open(file,'r').readlines():
        name=x.rstrip()
        bpm=pf.open(name)[0].header['BPM']
        new=name.split('/')[-1]
        root='/'.join(name.split('/')[:-1])
        shutil.copy(name,'./'+new)
        shutil.copy(root+'/'+bpm,'./'+bpm)
        out.write(new+'\n')
    out.close()

#--------------------------------------------------------------------------------------------------------
def bkgsub(file):

    data=pf.open(file)[0].data
    md=mode(data[np.where(data > 0)],axis=None)
    iraf.unlearn('imarith')
    iraf.imarith(file,'-',md[0][0],file[:-5]+'_bkgsub.fits')

#--------------------------------------------------------------------------------------------------------
def swarpHead(file,outfile):

    keys=['CRVAL1','CRVAL2','CRPIX1','CRPIX2','CDELT1','CDELT2',
          'CD1_1','CD1_2','CD2_1','CD2_2','NAXIS','NAXIS1','NAXIS2']
    head=pf.open(file)[0].header
    out=open(outfile,'w')
    for x in keys:
        out.write(x+' = '+str(head[x])+'\n')
    out.write('END     ')
    out.close()

#--------------------------------------------------------------------------------------------------------
def matchScales(inList,catFile='scale.cat',buff=1000.):

    (x,y,ra,dec,mag,star,flag)=np.loadtxt(catFile,usecols=(0,1,2,3,4,5,6),unpack=True,comments='#')
    cat=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
    idx,d2d,_=cat.match_to_catalog_sky(cat,2)
    (xmin,xmax,ymin,ymax)=(np.min(x),np.max(x),np.min(y),np.max(y))
    use=np.where((mag > 17.0) & (mag < 23.0) & (d2d.arcsec > 10.0) & (star > 0.9) & (flag == 0) &
                 (x > xmin + buff) & (x < xmax - buff) & (y > ymin + buff) & (y < ymax - buff))
    if len(use[0]) > 5000:
        good=use[0][:5000]
    else:
        good=use[0]
    (ra,dec)=(ra[good],dec[good])
    if os.path.exists('radec.dat'):
        os.remove('radec.dat')
    out=open('radec.dat','w')
    for x in range(len(ra)):
        out.write(str(ra[x]/15.0)+'\t'+str(dec[x])+'\n')
    out.close()

    iraf.unlearn('mscimatch')
    iraf.mscimatch('@'+inList,'radec.dat',scale='yes',zero='yes',box1='50',
                   box2='100',bpm='BPM',lower='-0.5',inter='yes',upper='200',niter='5',sigma='3.0')

#--------------------------------------------------------------------------------------------------------
def mkWeightMap(image,search=''):

    if search != '':
        imname=glob.glob(search)[0]
    else:
        imname=image
        
    iraf.unlearn('imcalc')
    iraf.imcalc(imname,'weight.fits','if im1 .eq. 0 then 0.0 else 1.0')

#--------------------------------------------------------------------------------------------------------
def subBack(image,bkg):

    iraf.unlearn('imcalc')
    if os.path.exists(image[:-5]+'_bkgsub.fits'):
        os.remove(image[:-5]+'_bkgsub.fits')
    iraf.imcalc(image+','+bkg,image[:-5]+'_bkgsub.fits','im1-im2')

#--------------------------------------------------------------------------------------------------------
def mkScaleImage(image):

    hdr=pf.open(image)[0].header
    (mult,add)=(hdr['MSCSCALE'],hdr['MSCZERO'])
    iraf.unlearn('imcalc')
    if os.path.exists(image[:-5]+'_scale.fits'):
        os.remove(image[:-5]+'_scale.fits')
    iraf.imcalc(image,image[:-5]+'_scale.fits','(im1+'+str(add)+')*'+str(mult))

#--------------------------------------------------------------------------------------------------------
def main(file):

    print '\n===WELCOME TO MOSAIC_COMBINE==='

    stackOut=file.split('.')[0]+'_stack.fits'
    l=open(file,'r').readlines()
    
    for x in l:
        singleFrame(x.rstrip())

    scaleToSingleZP()
    
#--------------------------------------------------------------------------------------------------------
def part2(file,outname):        
    finalCombine(outname)

