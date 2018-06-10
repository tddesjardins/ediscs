import os


def phot_script(img_dict, detection_img='rimg', outfile=None, clobber=False):
    """
    PURPOSE
    -------
    Generate the commands necessary to run Source Extractor on the input images for a given cluster.
    When the commands are run using Source Extractor, a background subtracted image will be
    generated for each input image with the suffix "_bkgsub.fits" appended to it. For the detection
    image, a segmentation image will also be generated with the suffix "_segmap.fits" appended.

    INPUTS
    ------
    img_dict (dict):
        Python dictionary of the images for a particular cluster.

    detection_img (str):
        Key for the detection image in img_dict to be used with Source Extractor's dual-image mode.
        If the key is missing from img_dict, a KeyError will be raised. Default is "rimg" for the
        R-band image. Note that the names of the dictionary keys are technically arbitrary, though
        values that make sense should be used.

    outfile (str):
        If specified, save the Source Extractor commands to a file. Default is None.

    clobber (bool):
        Overwrite outfile if it exists. Default is False.

    OUTPUTS
    -------
    phot_string (str):
        The string containing the Source Extractor commands for all input images.
    """

    if detection_img not in img_dict:
        raise KeyError('Detection image missing! Could not find key "{}" in input dictionary.'.format(detection_img))

    phot_string = []

    for filt in img_dict.keys():

        root_name = img_dict[filt].split('.')[0]
        catalog_name = root_name + '.cat'

        if filt is detection_img:
            out_img_types = '-BACKGROUND,SEGMENTATION'
            out_img_names = '{}_bkgsub.fits, {}_segmap.fits'.format(root_name, root_name)
        else:
            out_img_types = '-BACKGROUND'
            out_img_names = '{}_bkgsub.fits'.format(root_name)

        phot_string.append(('sex -c ediscs.sex -BACKPHOTO_TYPE "GLOBAL" -CATALOG_NAME {} -CHECKIMAGE_TYPE "{}" '
                            '-CHECKIMAGE_NAME "{}" {} {}').format(catalog_name, out_img_types, out_img_names,
                                                                  img_dict[detection_img], img_dict[filt]))

    if outfile is not None:
        if os.path.exists(outfile):
            if clobber:
                os.remove(outfile)
            else:
                raise OSError('{} already exists and clobber is False.'.format(outfile))
        out = open(outfile, 'w')
        for x in phot_string:
            out.write(x + '\n')
        out.close()

    return phot_string
