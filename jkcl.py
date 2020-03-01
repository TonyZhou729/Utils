import numpy as np
import healpy as hp
import time

class jkcl:
    # Initialization of jackknife procedure object.
    def __init__(self, nside=2048, lmax=2048, nside_jk=4):
        self.nside = nside # nside of real space maps. Default 2048
        self.lmax = lmax # lmax of output Cls and alm maps. Default 2048

        self.njk = hp.nside2npix(nside_jk) # Number of jacknifes = 12 * jacknife_nside^2
        # The larger this nside, the more jackknifes will be computed

        self.fsky = 1. - 1/self.njk # Area normalization based on jackknife.

        ### Cross Correlation data. Need to be initialized by calling the loadfiles method.
        self.map1 = None
        self.map2 = None
        
        self.result = np.arange(0, lmax+1, 1)
    
    def loadfiles(self, path1, path2=None, alm=False):
        '''
        Load data from file names using hp.read_...
        Assumes real map paths by default. If paths are alm maps, specify alm=True.
        If path2 is not specified, computed will be self correlation instead of cross correlation.
        '''
        if alm:
            self.map1 = hp.alm2map(hp.read_alm(path1), nside=self.nside)
            if path2 != None:
                self.map2 = hp.alm2map(hp.read_alm(path2), nside=self.nside)

        else:
            self.map1 = hp.ud_grade(hp.read_map(path1), nside_out=self.nside)
            if path2 != None:
                self.map2 = hp.ud_grade(hp.read_map(path2), nside_out=self.nside)

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
        Iterates through the njk jackknifes, computing a gross cross correlation each time
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
                print('Two maps are here.')
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








