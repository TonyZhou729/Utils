import numpy as np
import healpy as hp
import time

class jkcl:
    # Initialization of jackknife procedure object.
    def __init__(self, map1, map2=None, nside=2048, lmax=2048, nside_jk=4):
        '''
        Parameters:
        -----------
        map1: REAL SPACE Healpix map.
        map2: REAL SPACE Healpix map, or None.
            - Compute cross correlation if given, otherwise just uses the first map.
        nside: nside of real space maps. Default 2048.
        lmax: lmax of output Cls and alms used for computation. Default 2048
        nside_jk: nside of jackknife portion map, whose pixels will be used for masking.
            - If 4, compute 12*4*4 = 192 Jackknifes.
            - If 2, compute 12*2*2 = 48 Jackknifes.
            - Default is 4.
        '''

        self.nside = nside
        self.lmax = lmax

        self.njk = hp.nside2npix(nside_jk) # Number of jacknifes = 12 * jacknife_nside^2
        # The larger this nside, the more jackknifes to compute.

        self.fsky = 1. - 1/self.njk # Area normalization based on jackknife.

        ### Cross Correlation maps.
        self.map1 = hp.ud_grade(map1, nside_out=self.nside)
        self.map2 = (hp.ud_grade(map2, nside_out=self.nside) if map2 is not None else None)
        
        self.result = np.arange(0, lmax+1, 1)

    def pixel_mask(self, idx):
        '''
        Computes jackknife mask where the pixel indexed 'idx' is set to zero. The size of this
        pixel is 1/njk of the total map area. 
        We then ud_grade the size of the mask to match the correct nside.
        '''
        mask = np.ones(self.njk)
        mask[idx] = 0.
        mask = hp.ud_grade(mask, self.nside)
        return mask

    def compute_jkf(self):
        '''
        Iterates through the njk jackknifes, computing a cross correlation each time
        with 1/njk of the area taken out (from both maps). 
        Adds the resulting jackknifed cl to the total results using np.column_stack.
        There will be njk columns of cls, each from 0 to lmax.
        '''
    
        for idx in range(self.njk):
            print('Current masked index: %d/%d' % (idx+1, self.njk))
            mask = self.pixel_mask(idx)
            map1_masked = self.map1 * mask
            if self.map2 is not None:
                # Computes cross correlation if two maps are specified.
                map2_masked = self.map2 * mask
                cl = hp.anafast(map1_masked, map2_masked, lmax=self.lmax) / self.fsky
            else:
                # Simply self correlation if only one map is specified.
                cl = hp.anafast(map1_masked, lmax=self.lmax) / self.fsky
            
            self.result = np.column_stack((self.result, cl))

    def save_result(self, filename):
        '''
        Uses np.savetxt to save the resampled cls. 
        File will be stored with name 'filename', which must end in .txt.
        You may also include a path with this filename if you would like to save it
        to another directory.
        ''' 
        header = 'ell      cls' # lmax+1 ells, njk number of cls per ell.
        np.savetxt(filename, self.result, header=header, fmt='%5d'+' %15.7e'*self.njk)








