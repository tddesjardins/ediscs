import pandas as pd
from astropy.table import Table
import pdb,os
import matplotlib.pyplot as plt
import threedhst.eazyPy as eazy
import numpy as np

flux={'cB':'fB3','kB':'fB3','V':'fV3','R':'fR3','I':'fI3','z':'fz3','K':'fK3'}
err={'cB':'fB3err','kB':'fB3err','V':'fV3err','R':'fR3err','I':'fI3err','z':'fz3err','K':'fK3err'}
fabbrv={'cB':'F1','kB':'F2','V':'F3','R':'F4','I':'F5','z':'F6','K':'F7'}
eabbrv={'cB':'E1','kB':'E2','V':'E3','R':'E4','I':'E5','z':'E6','K':'E7'}

plt.rc('text', usetex=True)
plt.rc('font', family='palatino')

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
#    axes[0].scatter(lci, obs_sed, color='orange', zorder=2)
#    axes[0].scatter(lci, fobs, color='green', marker='s', s=150, zorder=2)

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


def prep(catalog,output,filters,defFile='zphot.translate',synthR=False,maglim=25.0,uppercut=17.0,starcut=True,addErr=0.0):

    if os.path.exists(output):
        os.remove(output)
        
    a=open(output,'a')
    obsData=Table.read(catalog).to_pandas()
    
    (fluxcols,errcols,cols)=([],[],[])
    for x in filters:
        cols.append(flux[x])
        fluxcols.append(flux[x])
        cols.append(err[x])
        errcols.append(err[x])
        
    a.write('#'+' '.join(cols)+'\n')

    if addErr > 0.0:
        for x in range(len(errcols)):
            obsData[errcols[x]] = np.sqrt((obsData[errcols[x]]**2.) + ((addErr*(obsData[fluxcols[x]]))**2.))
    
    if synthR == True:
        obsData['fracErr']=obsData['fR3err']/obsData['fR3']
        obsData.loc[obsData['fracErr'] > 0.1,'fR3err'] = obsData['fR3']*0.1
        
    obsData['rmag']=-2.5*np.log10(obsData['fR3'])+23.9
    for x in fluxcols:
        obsData.loc[obsData['rmag'] > maglim,x] = -99
        obsData.loc[flux2mag(obsData[x]) < uppercut,x] = -99
        if starcut == True:
            obsData.loc[obsData['starR'] > 0.1,x] = -99
        
    obsData.to_csv(a,header=False,index=False,sep='\t',columns=cols,na_rep='-99')
    a.close()
    if os.path.exists(defFile):
        os.remove(defFile)
    out=open(defFile,'w')
    for x in filters:
        out.write(flux[x]+' '+fabbrv[x]+'\n'+err[x]+' '+eabbrv[x]+'\n')
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
            
