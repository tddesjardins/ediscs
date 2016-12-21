import subprocess, os, pdb, shutil
from astropy.table import Table
import pandas as pd, numpy as np, eazy as e, matplotlib.pyplot as plt
from matplotlib import cm

#Just getting rid of some annoying clutter in the output
import warnings
warnings.simplefilter('ignore')

#Standard plotting settings
plt.rc('text', usetex=True)
plt.rc('font', family='palatino')

def plotResult(file='zpAdjust.ref',s=250,stepSize=0.02,xmag='V',ymag='R'):

    
    data=pd.read_table(file,delim_whitespace=True,header=None,
                       names=['iter','voff','roff','ioff','sigma','line'],
                       index_col=None,comment='#',
                       usecols=[0,1,2,3,4,5])

    colname=xmag.lower()+ymag.lower()
    slicemag={'vr':'i','rv':'i','ri':'v','ir':'v','vi':'r','iv':'r'}
    ui=np.sort(np.unique(data[slicemag[colname].lower()+'off']))

    for i in range(len(ui)):

        subdata=data.query(slicemag[colname].lower()+'off == '+str(ui[i]))
        
        sc=plt.scatter(subdata[xmag.lower()+'off'],subdata[ymag.lower()+'off'],
                       c=subdata['line'],marker='s',
                       s=s,cmap=cm.rainbow,vmin=np.min(data['line']),vmax=1.7)
        plt.colorbar(sc)
        plt.axis([np.min(subdata[xmag.lower()+'off'])-stepSize,
                  np.max(subdata[xmag.lower()+'off'])+stepSize,
                  np.min(subdata[ymag.lower()+'off'])-stepSize,
                  np.max(subdata[ymag.lower()+'off'])+stepSize])
        plt.xlabel(xmag.upper()+' offset')
        plt.ylabel(ymag.upper()+' offset')
        plt.title(slicemag[colname].upper()+' offset = '+str(ui[i]))
        if ui[i] < 0.:
            lab='m'+str(np.abs(ui[i]))
        else:
            lab='p'+str(ui[i])
        plt.savefig(colname+'_offsets_'+slicemag[colname].lower()+lab+'.pdf',format='pdf',dpi=6000)

        plt.clf()

    return data
    
def mag2fluxOffset(magoff):

    return 10.**(-0.4*magoff)

def main(fitsCat,maxoff=1.0,offrange=[],stepSize=0.05,fixV=-99.,fixR=-99.,fixI=-99.,maxV=20.,maxR=20.,maxI=20.,
         minV=-20.,minR=-20.,minI=-20.,outfile='zpAdjust.ref',plots=True):
    
    #make grid of ZP offsets
    if len(offrange) < 2:
        nsteps=np.round(((2.*maxoff)/stepSize))+1
        offsets=np.linspace(-1.*maxoff,maxoff,nsteps)
    else:
        nsteps=np.round(((offrange[1]-offrange[0])/stepSize))+1
        offsets=np.linspace(offrange[0],offrange[1],nsteps)
    (voff,roff,ioff)=(offsets[np.where((offsets >= minV) & (offsets <= maxV))],
                      offsets[np.where((offsets >= minR) & (offsets <= maxR))],
                      offsets[np.where((offsets >= minI) & (offsets <= maxI))])

    if fixV > -99.:
        voff=np.array([fixV])
    if fixR > -99.:
        roff=np.array([fixR])
    if fixI > -99.:
        ioff=np.array([fixI])
    
    filts={'V':voff,'R':roff,'I':ioff}

    outTable=open(outfile,'w')
    outTable.write('#Iter\tVoff\tRoff\tIoff\tScatter\tLineComp\n')

    x=0
    
    for i in range(len(voff)):
        for j in range(len(roff)):
            for k in range(len(ioff)):
                print 'Iteration '+str(x)+': Voff = '+str(voff[i])+', Roff = '+str(roff[j])+\
                  ', Ioff = '+str(ioff[k])+'\n'
                (vo,ro,io)=(mag2fluxOffset(voff[i]),mag2fluxOffset(roff[j]),mag2fluxOffset(ioff[k]))
                f=open('zphot.zeropoint','w')
                f.write('F3\t'+str(vo)+'\nF4\t'+str(ro)+'\nF5\t'+str(io))
                f.close()
                
                outTable.write('\t'.join([str(x),str(voff[i]),str(roff[j]),str(ioff[k])])+'\t')

                subprocess.Popen('/Users/tyler/eazy-photoz/src/eazy > logs/'+str(x)+'.log',shell=True).wait()

                (sz,line,_)=e.plotphotspec(fitsCat,selection='Rmag < 23. & Q5_7 == 4')

                if plots == True:
                    shutil.move('zspec_zphot.pdf','plots/zspec_zphot_'+str(x)+'.pdf')
                    shutil.move('photoz_residuals.pdf','plots/photoz_residuals_'+str(x)+'.pdf')

                outTable.write(str(sz)+'\t'+str(line)+'\n')

                outTable.flush()
                
                x += 1

    outTable.close()

    

    
