import pandas as pd
from astropy.table import Table
import pdb,os
import matplotlib.pyplot as plt
import threedhst.eazyPy as eazy
import numpy as np
from astropy.stats.funcs import biweight_location as bl

flux={'cB':'fB1','kB':'fB1','V':'fV1','R':'fR1','I':'fI1','z':'fz1','K':'fK1'}
corrflux={'cB':'cfB1','kB':'cfB1','V':'cfV1','R':'cfR1','I':'cfI1','z':'cfz1','K':'cfK1'}
err={'cB':'fB1err','kB':'fB1err','V':'fV1err','R':'fR1err','I':'fI1err','z':'fz1err','K':'fK1err'}
fabbrv={'cB':'F1','kB':'F2','V':'F3','R':'F4','I':'F5','z':'F6','K':'F7'}
eabbrv={'cB':'E1','kB':'E2','V':'E3','R':'E4','I':'E5','z':'E6','K':'E7'}
klam={'cB':4.125,'V':3.060,'R':2.548,'I':1.993,'z':1.384,'K':0.367}

plt.rc('text', usetex=True)
plt.rc('font', family='palatino')

def dered(filter, flux, ebv):

    ext = klam[filter] * ebv
    drflux = flux / (10.**(-0.4*ext))
    
    return drflux

def plotphotspec(catalog,zphot='OUTPUT/photz.zout',output='zspec_zphot.pdf',selection='',savetxt=False):

    if selection == '':
        selection='zLDP < 2.0'
    else:
        selection=selection+' & zLDP < 2.0'
    
    obsData=Table.read(catalog).to_pandas()
    index=range(len(obsData))
    obsData['index']=index
    obsData['Rmag']=-2.5*np.log10(obsData['fRauto'])+23.9
    trimData=obsData.query(selection)
    
    z=eazy.catIO.Readfile(zphot)
    trimData['zphot']=z.z_p[trimData['index']].tolist()

    #Zphot vs. ZLDP plot
    plt.plot([-10,10],[-10,10],'k-')
    plt.plot(trimData['zLDP'],trimData['zphot'],'ro')

    zmax=np.max(trimData['zLDP'])
    plt.axis([0,zmax,0,zmax])
    plt.xlabel(r'$z_{LDP}$')
    plt.ylabel(r'$z_{peak}$')

    plt.savefig(output, format='pdf', dpi=6000)
    plt.clf()

    plt.plot([-10,10],[0,0],'k-')
    residuals=trimData['zphot']-trimData['zLDP']
    plt.plot(trimData['zLDP'],residuals,'ro')
    plt.xlabel(r'$z_{LDP}$')
    plt.ylabel(r'$z_{peak}-z_{LDP}$')
    plt.plot([0.3,0.3],[-10,10],'k--')
    plt.plot([0.9,0.9],[-10,10],'k--')
    mean=bl(residuals.values[np.where((trimData['zLDP'].values >= 0.3) & (trimData['zLDP'].values <= 0.9))])
    plt.axis([0,zmax,-1.5,1.5])
    plt.text(0.4,0.75,'mean = '+str(np.around(mean,3)))

    plt.savefig('photoz_residuals.pdf', format='pdf', dpi=6000)
    plt.clf()

    trimData['residual']=trimData['zphot']-trimData['zLDP']

    trimData.to_csv('eazy_output.cat',index_label=False,sep='\t')
    
    pdb.set_trace()

def flux2mag(flux):

    return -2.5*np.log10(flux)+23.9

def plotResult(idnum,catalog,save=False):

    id=str(idnum)

    obsData=Table.read(catalog).to_pandas()
    rmag=-2.5*np.log10(obsData['fR3'][idnum])+23.9
    
    axes = eazy.plotExampleSED(idx=idnum, writePNG=False, MAIN_OUTPUT_FILE='photz',
                            OUTPUT_DIRECTORY='OUTPUT', CACHE_FILE='Same', lrange=[3800, 3.e4],
                            axes=None, individual_templates=False, fnu=False)
			   
    sed = eazy.getEazySED(idnum, MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT',
                        CACHE_FILE='Same', scale_flambda=True, verbose=False,
                        individual_templates=False)

    lambdaz, temp_sed, lci, obs_sed, fobs, efobs = sed

    zgrid, pzi, prior = eazy.getEazyPz(idnum, MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT',
                                    CACHE_FILE='Same', binaries=None, get_prior=True)
    
    axes[1].plot(zgrid, pzi, color='black')
    axes[1].plot(zgrid, prior/prior.max()*pzi.max(), color='purple')

    axes[1].set_xlim([0.,1.])
    axes[0].text(5500.,axes[0].get_ylim()[1]*0.1,r'$m_{R}$ = '+str(rmag)[:5]+' mag')

    if save == True:
        plt.savefig('OUTPUT/'+id+'.pdf',format='pdf',dpi=6000)
        plt.clf()
    else:
        plt.show()
    plt.close()


def prep(catalog,output,filters,defFile='zphot.translate',synthR=False,maglim=99.0,uppercut=0.0,starcut=False):

    if os.path.exists(output):
        os.remove(output)
        
    a=open(output,'a')
    obsData=Table.read(catalog).to_pandas()
    
    (fluxcols,errcols,cols)=([],[],[])
    for x in filters:
        cols.append(corrflux[x])
        fluxcols.append(corrflux[x])
        cols.append(err[x])
        errcols.append(err[x])
    cols.append('fRauto')
    fluxcols.append('fRauto')
        
    a.write('#'+' '.join(cols)+'\n')
    
    if synthR == True:
        obsData['fracErr']=obsData['fR1err']/obsData['fR1']
        obsData.loc[obsData['fracErr'] > 0.1,'fR1err'] = obsData['fR1']*0.1
        
    obsData['rmag']=-2.5*np.log10(obsData['fR1'])+23.9
    for x in fluxcols:
        obsData.loc[obsData['rmag'] > maglim,x] = -99
        obsData.loc[flux2mag(obsData[x]) < uppercut,x] = -99
        if starcut == True:
            obsData.loc[obsData['starR'] > 0.1,x] = -99

    for x in filters:
        obsData[corrflux[x]] = dered(x, obsData[flux[x]], obsData['ebv'])
#    pdb.set_trace()

    obsData['cfRauto']=dered('R',obsData['fRauto'],obsData['ebv'])
    obsData.to_csv(a,header=False,index=False,sep='\t',columns=cols,na_rep='-99')
    a.close()
    if os.path.exists(defFile):
        os.remove(defFile)
    out=open(defFile,'w')
    for x in filters:
        out.write(corrflux[x]+' '+fabbrv[x]+'\n'+err[x]+' '+eabbrv[x]+'\n')
    out.write('cfRauto TOT4\n')
    out.close()
        
def main(redfile='OUTPUT/photz.zout',catalog=''):

    df=pd.read_table(redfile,delim_whitespace=True,
                    header=None,names=['z'],comment='#',
                    index_col=None,usecols=[2])
    res = np.where(df['z'] != -99)
    print str(len(res[0]))+' good measurements!'
    for x in range(len(df)):
        if df['z'][x] != -99:
            plotResult(x,catalog,save=True)
            
