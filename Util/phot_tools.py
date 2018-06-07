

def photscript(listfile, clname, photfile='photscript'):
    """
    PURPOSE: Generate a script (to be called on with source from the command line) that will run SExtractor on
    \tall input images in dual-image mode (using the R-band for detection) for a cluster.

    INPUTS:
    \tlistfile - two columns of ?img (where ? is the filter letter) and image file names
    \tclname   - name of the cluster (e.g., cl1354)
    \tphotfile - name of the output script file (optional)

    RETURNS: None.
    """

    (keys, files) = np.loadtxt(listfile, usecols=(0, 1), unpack=True,
                               dtype={'names': ('keys', 'files'), 'formats': ('S5', 'S30')})
    imgs = {}
    for x in range(len(keys)):
        imgs[keys[x]] = files[x]

    outfile = open('photscript', 'w')

    string1 = 'sex -c ediscs.sex -BACKPHOTO_TYPE "GLOBAL" -CATALOG_NAME '
    string2 = ' -CHECKIMAGE_TYPE "-BACKGROUND" -CHECKIMAGE_NAME '
    finalString = ''

    finalString = finalString + string1 + clname + '_r.cat' + ' -CHECKIMAGE_TYPE "-BACKGROUND,SEGMENTATION" -CHECKIMAGE_NAME "' \
                  + imgs['rimg'][:-5] + '_bkgsub.fits' + ',' + imgs['rimg'][:-5] + '_segmap.fits" ' + imgs[
                      'rimg'] + ',' + imgs['rimg'] + '\n'

    if 'bimg' in keys:
        finalString = finalString + string1 + clname + '_b.cat' + string2 + imgs['bimg'][:-5] + '_bkgsub.fits ' + imgs[
            'rimg'] + ',' + imgs['bimg'] + '\n'
    if 'vimg' in keys:
        finalString = finalString + string1 + clname + '_v.cat' + string2 + imgs['vimg'][:-5] + '_bkgsub.fits ' + imgs[
            'rimg'] + ',' + imgs['vimg'] + '\n'
    if 'iimg' in keys:
        finalString = finalString + string1 + clname + '_i.cat' + string2 + imgs['iimg'][:-5] + '_bkgsub.fits ' + imgs[
            'rimg'] + ',' + imgs['iimg'] + '\n'
    if 'zimg' in keys:
        finalString = finalString + string1 + clname + '_z.cat' + string2 + imgs['zimg'][:-5] + '_bkgsub.fits ' + imgs[
            'rimg'] + ',' + imgs['zimg'] + '\n'
    if 'kimg' in keys:
        finalString = finalString + string1 + clname + '_k.cat' + string2 + imgs['kimg'][:-5] + '_bkgsub.fits ' + imgs[
            'rimg'] + ',' + imgs['kimg'] + '\n'
    if 'rbimg' in keys:
        finalString = finalString + string1 + clname + '_rb.cat' + string2 + imgs['rbimg'][:-5] + '_bkgsub.fits ' + \
                      imgs['rimg'] + ',' + imgs['rbimg'] + '\n'
    if 'rkimg' in keys:
        finalString = finalString + string1 + clname + '_rk.cat' + string2 + imgs['rkimg'][:-5] + '_bkgsub.fits ' + \
                      imgs['rimg'] + ',' + imgs['rkimg'] + '\n'

    out = open(photfile, 'w')
    out.write(finalString)
    out.close()