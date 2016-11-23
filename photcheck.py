import numpy as np, pdb, matplotlib.pyplot as plt, os, pyfits as pf, pdb
import matplotlib.cm as cm
from astropy import units as u
from astropy import coordinates
import matplotlib.patches as patches
from astropy.stats.funcs import biweight_location as bl
from matplotlib.colors import LogNorm


sep='\t-----------------'
plt.rc('text', usetex=True)
plt.rc('font', family='palatino')

fcolors='/Users/tyler/Documents/cmdplots/forsdata/fors_colors.dat'

def matchfors(ra, dec, forsra, forsdec, tolerance=1.0):

    forsCat=coordinates.SkyCoord(ra=forsra*u.degree, dec=forsdec*u.degree)
    wfiCat=coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    idx, d2d, _ = wfiCat.match_to_catalog_sky(forsCat)
    _, n2, _ = wfiCat.match_to_catalog_sky(wfiCat,2)
    match = np.where(d2d.arcsec <= tolerance)

    return (idx, match, d2d.arcsec, n2.arcsec)

def abErr(flux,err):

    return 1.085736*(err/flux)

def ujy2ab(flux):

    return -2.5*np.log10(flux)+23.9

def splitCoords(x,y,n):

    if type(n) is not float:
        n=float(n)
    
    section=[]
    #The +1 fudge factor stops the max Y and max X points from being
    #in their own special bins
    (divx,divy)=((np.max(x)+1.)/n,(np.max(y)+1.)/n)

    print '\tSplitting '+str(len(x))+' coordinates into a '\
      +str(int(n))+'x'+str(int(n))+' grid'
    print '\tMax X: '+str(np.max(x))+'; Max Y: '+str(np.max(y))
    print '\tX Grid Spacing: '+str(divx)+'; Y Grid Spacing: '+str(divy)

    #Divide up the FOV into a 4x4 grid increasing from left to right
    #and bottom to top
    for i in range(len(x)):
        section.append('('+str(int(np.floor(x[i]/divx)))+','+
                        str(int(np.floor(y[i]/divy)))+')')

    return np.array(section),divx,divy

def plotGridCMD(ra,dec,v0,r0,i0,verr0,rerr0,ierr0,gridSecs0,sexclass,saveFig='',clname='',fwhm0=[0],xs=0,ys=0,xcdt=[0],ycdt=[0],
                magrange=[23.5,18.5],classStar=0.99):

    #Bring in the FORS data
    (wmin, starflag) = np.loadtxt('VLT.starflag.dat', usecols=(0, 1), unpack = True, comments = '#')
    (forsRA, forsDec, forsV, forsR, forsI) = np.loadtxt('fors.dat', usecols = (1, 2, 9, 10, 11),
                                                        unpack = True, comments = '#')
     
    #Match to FORS RA and Dec
    goodfors = np.where((wmin > 0.2) & (starflag == 1) & (forsR < 23.0) & (forsR > 17.0)
                          & (forsV > 17.0) & (forsI > 17.0))
    (sforsRA, sforsDec, sforsV, sforsR, sforsI) = (forsRA[goodfors] * 15.0, forsDec[goodfors],
                                                   forsV[goodfors] + 0.01850, forsR[goodfors] + 0.19895,
                                                   forsI[goodfors] + 0.42143)
    (idx, _, d2d, n2) = matchfors(ra, dec, sforsRA, sforsDec, tolerance=1.0)

    match = np.where((r0 >= magrange[1]) & (r0 <= magrange[0]) & (d2d <= 1.0) & (n2 > 3.0))

    stars=np.where((sexclass >= classStar) & (r0 <= magrange[0]) & (r0 >= magrange[1]) & (n2 > 3.0))[0]


    (v,r,i,verr,rerr,ierr,gridSecs)=(v0[stars],r0[stars],i0[stars],verr0[stars],rerr0[stars],ierr0[stars],gridSecs0[stars])

    print '\n\tComputing colors for CMD check plot...'
    (cmdx,cmdy)=(v-r, r-i)

    good=np.where((cmdy > -0.5) & (cmdy < 2.0) & (cmdx > -0.5) & (cmdx < 1.5))

    meanxerr=bl(np.sqrt((verr[good]**2.0)+(rerr[good]**2.0)))
    meanyerr=bl(np.sqrt((rerr[good]**2.0)+(ierr[good]**2.0)))

    #Plot different parts of the grid in different colors
    uniq=np.unique(gridSecs,return_index=True,return_counts=True)
    (nGrid,uGrid)=(len(uniq[1]),[])
    for x in range(nGrid):
        uGrid.append(gridSecs[uniq[1][x]])
    colorList=cm.rainbow(np.linspace(0,1,nGrid))

    print '\tPlotting CMDs...\n'
    for i in range(nGrid):
        use=np.where(np.char.strip(gridSecs)==uGrid[i])
        color=[]
        for j in range(len(use[0])):
            color.append(colorList[i])
        plt.scatter(cmdx[use],cmdy[use],color=color,s=4)#,cmap=cm.rainbow)

    #Plot ALL of the FORS stars in the background
#    (afv,afr,afi)=np.loadtxt(fcolors,usecols=(0,1,2),unpack=True)
#    plt.scatter(afv-afr,afr-afi,marker='o',s=100.,alpha=0.1,color='k')
#    (forsx,forsy)=(afv-afr,afr-afi)

     # calculate the 2D density of the data given
#    counts,xbins,ybins=np.histogram2d(forsx,forsy,bins=5)#,normed=LogNorm())
    # make the contour plot
#    plt.contourf(counts.transpose(),extent=[xbins.min(),xbins.max(),
#        ybins.min(),ybins.max()],colors='black',alpha=0.4,levels=[0,1,5,10,15,20,25])
#    plt.contour(counts,levels=[0,1,5,10,15])

#    pdb.set_trace()

    (forsx,forsy)=(sforsV[idx][match]-sforsR[idx][match],sforsR[idx][match]-sforsI[idx][match])
    plt.scatter(forsx,forsy,color='k',s=40.,marker='s',alpha=0.6)
    plt.text(-0.4,1.7,clname,fontsize=20,style='italic')
        
    plt.xlabel('$(V-R)$ / AB mag')
    plt.ylabel('$(R-I)$ / AB mag')
    plt.axis([-0.5,1.5,-0.5,2])
    plt.errorbar(1.2,-0.0,xerr=meanxerr,yerr=meanyerr,color='k')
    
    if saveFig == '':
        plt.show()
    else:
        if os.path.exists(saveFig):
            ans=raw_input('\tWARNING! '+saveFig+
                            ' already exists! Delete and proceed? ')
        else:
            ans='Y'
        if ans.upper() == 'Y':
            if os.path.exists(saveFig):
                os.remove(saveFig)
            plt.savefig(saveFig,format='pdf',dpi=6000)
            plt.close()
        else:
            print '\tNOTE: Plot will not be saved.'

    savfwhm=[]
    if len(fwhm0) > 2:
        fwhm=fwhm0[stars]
        for i in range(nGrid):
            use=np.where(np.char.strip(gridSecs)==uGrid[i])
            color=[]
            for j in range(len(use[0])):
                color.append(colorList[i])
            if len(fwhm[use]) > 1:
                meany=bl(fwhm[use])
            elif len(fwhm[use]) == 1:
                meany=fwhm[use][0]
            elif len(fwhm[use]) == 0:
                meany=0
            savfwhm.append(meany)
            plt.scatter(np.zeros(len(use[0]))+i+1,fwhm[use],color=color,s=16,marker='o',alpha=0.3)
            plt.scatter(i+1,meany,color='k',s=80,marker='*')

        fwhmplot=saveFig[:-4]+'_fwhm.pdf'
        if os.path.exists(fwhmplot):
            os.remove(fwhmplot)
        plt.ylabel('FWHM / arcsec')
        plt.xlabel('Quadrant')
        plt.text(1,2.2,clname,style='italic',fontsize=20)
        plt.axis([0,18,0.5,2.5])
        plt.savefig(fwhmplot,format='pdf',dpi=6000)
        plt.close()

        coords=uniq[0]
        numpts=uniq[2]

        ax1=plt.axes()
#        ax1=fig1.add_subplot(111,aspect='equal')
        for i in range(len(coords)):
            corner=(np.min(xcdt)+(xs*float(coords[i][1])),np.min(ycdt)+float(ys*float(coords[i][3])))
            ax1.add_patch(patches.Rectangle(corner,xs,ys,color=colorList[i],alpha=0.4))
            plt.text(corner[0]+(0.3*xs),corner[1]+(0.35*ys),'Section: '+str(i+1)+'\nN Stars: '+str(numpts[i])+'\n'+str(np.around(savfwhm[i],2))+'"',fontsize=12)

        pplot=saveFig[:-4]+'_regions.pdf'
        if os.path.exists(pplot):
            os.remove(pplot)
        plt.ylabel('Y')
        plt.xlabel('X')
        if np.min(xcdt) < 200.:
            minx = 0.
        else:
            minx = np.around(np.min(xcdt),-3)-200.
        if np.min(ycdt) < 200.:
            miny = 0.
        else:
            miny = np.around(np.min(ycdt),-3)-200.
        plt.axis([minx,minx+(4.2*xs),miny,miny+(4.2*ys)])
        plt.savefig(pplot,format='pdf',dpi=6000)
        plt.close()


def fakeData(N,s1=0.25,s2=0.35,m1=18.5,m2=18.4,maxx=4096.,maxy=4096.):

    if type(maxx) is not float:
        maxx=float(maxx)
    if type(maxy) is not float:
        maxy=float(maxy)
    out=open('test.data','w')
    id=np.array(range(N))+1
    x=np.random.rand(N)*maxx
    y=np.random.rand(N)*maxy
    mag=s1*np.random.randn(N)+m1
    mag2=s2*np.random.randn(N)+m2
    out.write('#ID\tX\tY\tMAG\n')
    for i in range(len(id)):
        out.write(str(id[i])+'\t'+str(x[i])+'\t'+str(y[i])+
                  '\t'+str(mag[i])+'\t'+str(mag2[i])+'\n')
    out.close()
            
def main(file,saveFig='',classStar=0.99,magrange=[22.,18.],clname=''):

    print '\n'+sep+'\n'

    if saveFig=='':
        saveFig=file[:-12]+'stellar.pdf'
    if clname=='':
        clname=file[:6]
    
    hdu=pf.open(file)
    data=hdu[1].data
    (gridSecs,spacex,spacey)=np.array(splitCoords(data['x'],data['y'],4))
    (ra,dec)=(data['ra'],data['dec'])
    (v,r,i)=(ujy2ab(data['fV3']),ujy2ab(data['fR3']),ujy2ab(data['fI3']))
    (verr,rerr,ierr)=(abErr(data['fV3'],data['fV3err']),abErr(data['fR3'],data['fR3err']),abErr(data['fI3'],data['fI3err']))
    fwhmR=data['fwhmR']

    plotGridCMD(ra,dec,v,r,i,verr,rerr,ierr,gridSecs,data['starR'],saveFig=saveFig,
                clname=clname,fwhm0=fwhmR,xs=spacex,ys=spacey,xcdt=data['x'],ycdt=data['y'],magrange=magrange,classStar=classStar)

    print '\n\tDone!!!\n'
    print sep+'\n'

