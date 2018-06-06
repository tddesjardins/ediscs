class Constants(object):
    """
    This object will contain some useful constants the can be called on in
    other parts of the code. For example:

    >>> from Util import constants as const
    >>> med_cterms = const.Constants.med_cterms
    """

    def __init__(self):

        return None

    def med_cterms(self, key):

        med_cterms_arr = {'vr_v': -0.151, 'vr_r': 0.0245, 'vi_v': -0.0725, 'vi_i': 0.1465,
                          'ri_r': 0.015, 'ri_i': 0.238}

        try:
            value = med_cterms_arr[key.lower()]
        except KeyError:
            raise KeyError('Did not recognize key {} for med_cterms!'.format(key.lower()))

        return value

    def vega_to_ab(self, key):

        vega_to_ab_arr = {'bctio': -0.09949, 'bkpno': -0.10712, 'v': 0.01850, 'r': 0.19895,
                          'i': 0.42143, 'k': 1.84244}

        try:
            value = vega_to_ab_arr[key.lower()]
        except KeyError:
            raise KeyError('Did not recognize key {} for vega_to_ab!'.format(key.lower()))

        return vega_to_ab_arr
