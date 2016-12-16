import pyfits as pf
from pyraf import iraf
import numpy as np
from iraf import stsdas,imgtools,toolbox
import glob,subprocess,shlex,pdb,shutil,os

NewfirmZPoffset={'P550-Ca':0.071,'P550-Cb':0.043,'cl1054-12a':0.083,
                 'cl1054-12b':0.012,'cl1059a':0.092,'cl1059b':0.095,
                 'cl1138a':0.114,'cl1138b':0.103,'cl1227a':0.053,
                 'cl1227b':0.051,'cl1232':0.041,'cl1354':0.073,
                 'cl1420a':0.083,'cl1420b':0.081,'cl1054-12c':0.104,
                 'cl1018a':0.126,'cl1018b':0.113,'cl1354b':0.117,
                 'cl1301a':0.122,'cl1301b':0.125,'cl1232b':0.063,
                 'cl1420c':0.100,'cl1420d':0.138,'cl1420e':0.138,
                 'cl1411':0.052,'image0':0.054,'image1':0.053,'image2':0.037,
                 'image3':0.049,'image4':0.007}

NewfirmZPoffsig={'p550-ca':0.052,'p550-cb':0.054,'cl1054-12a':0.059,
                 'cl1054-12b':0.049,'cl1059a':0.051,'cl1059b':0.054,
                 'cl1138a':0.044,'cl1138b':0.048,'cl1227a':0.046,
                 'cl1227b':0.053,'cl1232':0.044,'cl1354':0.061,
                 'cl1420a':0.063,'cl1420b':0.075,'cl1054-12c':0.045,
                 'cl1018a':0.060,'cl1018b':0.069,'cl1354b':0.056,
                 'cl1301a':0.041,'cl1301b':0.040,'cl1232b':0.033,
                 'cl1420c':0.061,'cl1420d':0.062,'cl1420e':0.060,
                 'cl1411':0.057,'image0':0.106,'image1':0.102,'image2':0.064,
                 'image3':0.060,'image4':0.161}

def file_check(file,delete=False):
    if os.path.exists(file):
        res=True
    else:
        res=False

    if res==True:
        if delete==True:
            os.remove(file)

    return res
    
def get2masspsc(img):

    """
    """

    result=subprocess.Popen('imcat -c tmc -sm 12,14 '+img,shell=True,
                            stdout=subprocess.PIPE)
    cat=result.communicate()[0].split('\n')[:-1]
    return cat

def remap(img,ref):

    """
    """

    subprocess.Popen('remap -f '+ref+' -o remapped.fits '+img,
                     shell=True).wait()

def scalecounts(zpimg,zpref):

    """
    """

    result=10.0**(-0.4*(zpimg-zpref))
    return result

def classify(search):

    """
    """

    files=glob.glob(search)
#    print files
    for i in files:
        imtype=pf.open(i)[0].header['PRODTYPE']
        if 'image' in imtype:
            image=i
        elif 'dqmask' in imtype:
            dqmask=i
        elif 'expmap' in imtype:
            expmap=i

    return (image,dqmask,expmap)

def copyhead(img,ref='',ext=0,offset=0):

    """
    """
    
    header=pf.open(ref)[ext].header
    hdu=pf.open(img,mode='update')
    (cr1,cr2)=(header['CRPIX1'],header['CRPIX2'])
    hdu[0].header=header
    hdu[0].header['CRPIX1']=cr1+offset
    hdu[0].header['CRPIX2']=cr2+offset
    hdu[0].header['BPM']=img[:-5]+'_mask.pl'
    
    hdu.close()

def applywcs(input,output,cat='tmc'):

    """
    """

#    hdr=pf.open(ref)[0].header
#    (ra,dec)=(hdr['RA'],hdr['Dec'])

    file_check('tmp_wcs.fits',delete=True)
#    subprocess.Popen('imwcs -c '+cat+' -q ir -e -h 200 -t '+\
#                     '1000 -i -5 -n 4 -p 0.4 -o tmp_wcs.fits -wv -j '+\
#                     ra+' '+dec+' '+input,shell=True).wait()
    subprocess.Popen('imwcs -c '+cat+' -q ir -e -h 200 '+\
                     '-i -5 -n 4 -p 0.4 -o tmp_wcs.fits -wv '+\
                     input,shell=True).wait()


    file_check(output,delete=True)
    subprocess.Popen('imwcs -c '+cat+' -q irs -h 2000 -m 12 14 '+\
                     '-i -3 -n 4 -o '+output+' -wv tmp_wcs.fits',
                     shell=True).wait()

#    os.remove('tmp_wcs.fits')
    
def stacking(cllist,zpofflist,ref,zprefoff=0.0,stackname='stack',shiftsize=400):

    """
    """

    #Reset the IRAF tasks used in this routine.
    iraf.unlearn('imcalc')
    iraf.unlearn('imcombine')
    iraf.unlearn('imreplace')
    iraf.unlearn('xyxymatch')
    iraf.unlearn('geomap')
    iraf.unlearn('geotran')
    iraf.unlearn('imcopy')

    #Find reference image in reference directory. Check to make
    #sure that it is actually the image and not the mask file!
    #Grab the mask for adding to the mask list now.
    (refimg,refmask,expmap)=classify(ref+'/tu*.fits')
    zpref=pf.open(refimg)[0].header['MAGZERO']
#    zprefoff=NewfirmZPoffset[ref.split('/')[-1]]
    zprefoff=float(zprefoff)

    #Get 2MASS PSC positions for reference cluster image.
    catalog=get2masspsc(refimg)
    foo=file_check(ref+'/2mass_ref_stars.cdt',delete=True)
    foo=open(ref+'/2mass_ref_stars.cdt','w')
    for y in catalog:
        data=y.split()
        foo.write(data[6]+'\t'+data[7]+'\n')
    foo.close()

    #Create lists for files to be input into the stacking routine.
    foo=file_check('matchlist',delete=True)
    foo=file_check('scalelist',delete=True)
    foo=file_check('shiftlist',delete=True)
    foo=file_check('masklist',delete=True)
    foo=file_check('shiftmask',delete=True)
    foo=file_check('expmaplist',delete=True)
    (matchlist,scalelist,shiftlist,masklist,
     shiftmask,finalmasks,stacklist,stackmask,
     finalmasks2,expmaplist,shiftexp,expmaplist2)=(open('matchlist','w'),open('scalelist','w'),
                                        open('shiftlist','w'),open('masklist','w'),
                                        open('shiftmask','w'),open('finalmasks','w'),
                                        open('stacklist','w'),open('stackmask','w'),
                                        open('finalmasks2','w'),open('expmaplist','w'),
                                        open('shiftexp','w'),open('expmaplist2','w'))
    (xsize,ysize)=(np.array([]),np.array([]))
    
    
    #Step through all of the input cluster directories.
    i=0
    for x in cllist:
        #Find the image, mask, and exposure map files. Get zeropoints and
        #scale image to the reference image.
        scaleimg=x+'/scaled_to_'+ref.split('/')[-1]+'.fits'
        foo=file_check(scaleimg,delete=True)
        (img,mask,expmap)=classify(x+'/tu*.fits')
        imgzp=pf.open(img)[0].header['MAGZERO']
        (xs,ys)=(pf.open(img)[0].header['NAXIS1'],pf.open(img)[0].header['NAXIS2'])
        (xsize,ysize)=(np.append(xsize,xs),np.append(ysize,ys))

        imgzpoff=float(zpofflist[i])
#        imgzpoff=NewfirmZPoffset[x.split('/')[-1]]
        scale=scalecounts(imgzp+imgzpoff,zpref+zprefoff)
        iraf.imcalc(img,scaleimg,'im1*'+str(scale))

        #Get X,Y pixel positions of 2MASS sources from the 2MASS PSC
        #in the image. Use these to compute shifts relative to the
        #reference image using IRAF task geomap.
        foo=file_check(x+'/2mass_ref_stars.cdt',delete=True)
        foo=open(x+'/2mass_ref_stars.cdt','w')
        catalog=get2masspsc(scaleimg)
        for y in catalog:
            data=y.split()
            foo.write(data[6]+'\t'+data[7]+'\n')
        foo.close()
    
        #Match the 2MASS PSC positions with stars in the reference
        #image using xyxymatch. The matched source list is then fed
        #into geomap to get the X and Y shifts.
        foo=file_check(x+'/2mass_matched.cdt',delete=True)
        iraf.xyxymatch(x+'/2mass_ref_stars.cdt',ref+'/2mass_ref_stars.cdt',
                       x+'/2mass_matched.cdt','200.0',verbose='no')

        #Append all of the names of the files for the input and output filename
        #lists to be passed to IRAF tasks further down the line.
        matchlist.write(x+'/2mass_matched.cdt\n')
        scalelist.write(scaleimg+'\n')
        foo=file_check(x+'/scaled_and_shifted.fits',delete=True)
        shiftlist.write(x+'/scaled_and_shifted.fits['+str(shiftsize)+':'+\
                 str(int(np.max(xsize))+shiftsize)+','+str(shiftsize)+':'+\
                 str(int(np.max(ysize))+shiftsize)+']\n')
        stacklist.write(x+'/scaled_and_shifted.fits\n')
        file_check(x+'/mask_tmp.fits',delete=True)
        file_check(x+'/expmap_tmp.fits',delete=True)
        iraf.imarith(mask+'[1]','*',1000.0,x+'/mask_tmp.fits',pixtype='real')
        iraf.imarith(expmap+'[1]','*',1.0,x+'/expmap_tmp.fits',pixtype='real')
        offset=2.558435
        file_check(x+'/mask_tmp2.fits',delete=True)
        iraf.imcalc(x+'/mask_tmp.fits',x+'/mask_tmp2.fits','im1+'+str(offset))
        os.remove(x+'/mask_tmp.fits')
        masklist.write(x+'/mask_tmp2.fits\n')
        file_check(x+'/mask_shift.fits',delete=True)
        shiftmask.write(x+'/mask_shift.fits['+str(shiftsize)+':'+\
                    str(int(np.max(xsize))+shiftsize)+','+str(shiftsize)+':'+\
                    str(int(np.max(ysize))+shiftsize)+']\n')
        stackmask.write(x+'/mask_shift.fits\n')
        finalmasks.write(x+'/mask_final.fits\n')
        finalmasks2.write(x+'/mask_final.fits[0]\n')
        expmaplist.write(x+'/expmap_tmp.fits[0]\n')
        shiftexp.write(x+'/expmap_shift.fits['+str(shiftsize)+':'+\
                    str(int(np.max(xsize))+shiftsize)+','+str(shiftsize)+':'+\
                    str(int(np.max(ysize))+shiftsize)+']\n')
        expmaplist2.write(x+'/expmap_shift.fits\n')
        i += 1

    #Close all of the input and output filename lists to be passed to IRAF tasks.
    matchlist.close()
    scalelist.close()
    stacklist.close()
    masklist.close()
    shiftmask.close()
    finalmasks.close()
    shiftlist.close()
    stackmask.close()
    finalmasks2.close()
    expmaplist.close()
    expmaplist2.close()
    shiftexp.close()

    #Get the shifts between all input files (including the reference) and the
    #reference image itself.
    foo=file_check('shift.db',delete=True)
    iraf.geomap('@matchlist','shift.db',1.0,np.max(xsize),
                1.0,np.max(ysize),fitgeometry='shift',interactive='no',
                maxiter=2,function='legendre',verbose='no')

    #Shift the input images (including the reference) and associated mask files
    #to a common pixel grid. Add some padding around the individual frames (-99
    #in the images, 1 in the bad pixel masks) to ensure that the images will
    #combine properly.
    (maxx,maxy)=(np.max(xsize)+shiftsize+100.0,np.max(ysize)+shiftsize+100.0)
    iraf.geotran('@scalelist','@shiftlist','shift.db','@matchlist',geometry='linear',
                 boundary='constant',nlines=maxy,ncols=maxx,constant=-99.0)

    iraf.geotran('@masklist','@shiftmask','shift.db','@matchlist',geometry='linear',
                 boundary='constant',nlines=maxy,ncols=maxx,constant=1000.0,
                 nxblock=10000,nyblock=10000)
    
    iraf.geotran('@expmaplist','@shiftexp','shift.db','@matchlist',geometry='linear',
                 boundary='constant',nlines=maxy,ncols=maxx,constant=0.)

    for x in cllist:
        file_check(x+'/mask_final.fits',delete=True)
        shutil.copy(x+'/mask_shift.fits',x+'/mask_final.fits')
        iraf.hedit(x+'/scaled_and_shifted.fits[0]','BPM',x+'/mask_final.fits[0]',
                   add='yes',update='yes',verify='no')
    iraf.imreplace('@finalmasks2',0,upper=offset)
    iraf.imreplace('@finalmasks2',1,lower=offset)

    file_check(stackname,delete=True)
    file_check(stackname[:-5]+'_mask.pl',delete=True)
    file_check(stackname[:-5]+'_expmap.fits',delete=True)
    iraf.imcombine('@stacklist',stackname,bpmasks=stackname[:-5]+'_bpm',
                   masktype='goodval',reject='none',mclip='yes',lthresh='INDEF',hthresh='INDEF',
                   hsigma=10.0,lsigma='INDEF',nrejmasks=stackname[:-5]+'_nrej',
                   sigmas=stackname[:-5]+'_sigma',grow=2.5,nkeep=1,blank=-99.0,gain=8.0,rdnoise=35.0)
    iraf.imcombine('@expmaplist2',stackname[:-5]+'_expmap.fits',combine='sum')
    hdu=pf.open(stackname,mode='update')
    hdu[0].header['BPM']=stackname.split('/')[-1][:-5]+'_mask.pl'
    hdu[0].header['MAGZERO']=zpref+zprefoff
    hdu.close()

	#Fix the WCS information in the stacked image.
    copyhead(stackname,refimg,offset=shiftsize)
    applywcs(stackname,stackname[:-5]+'_wcs.fits')

    trash=['matchlist','scalelist','shiftlist','masklist','shiftmask','finalmasks',
            'shift.db','stacklist','finalmasks2','stackmask','tmp_wcs.fits','expmaplist',
            'expmaplist2','shiftexp']
    for x in trash:
        os.remove(x)

def readcontrol(file):

    print '\n==WELCOME TO IMSTACK=='

    (l,d)=(open(file,'r').readlines(),{})
    for x in l:
        data=x.rstrip().split(':')
        if ('Files' in data[0]) or ('ZPoffsets' in data[0]):
            d[data[0]]=''.join(data[1].split()).split(',')
        else:
            d[data[0]]=data[1].strip()
    expectedKeys=['StackName','Reference','ZPrefoffset','Files','ZPoffsets']

    print '\nUsing Following Parameters\n' \
        '--------------------------\n'
    for x in expectedKeys:
        try:
            if x not in ('Files','ZPoffsets'):
                print x+': '+d[x]
            else:
                print x+': '+', '.join(d[x])
        except KeyError:
            if x == 'ZPoffsets':
                print '\nWarning: No zeropoint offsets supplied for input files.\n' \
                    'Assuming all offsets to be 0.0 mag.\n'
                d[x]=np.zeros(len(d['Files'])).tolist()
            else:
                raise Warning('Input file is missing keyword: '+x)

    #Check if the number of ZP offsets and files matches
    if len(d['Files']) != len(d['ZPoffsets']):
        print 
        raise Warning('Number of files and zeropoint offsets do not match!')
        
    return d

def main(file):

    d=readcontrol(file)
    stacking(d['Files'],d['ZPoffsets'],d['Reference'],zprefoff=d['ZPrefoffset'],
            stackname=d['StackName'])
