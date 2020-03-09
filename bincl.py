import numpy as np
import os

# Initializes object for binning numerical arrays with large fluctuations.
# Useful for CMB power spectra and other signal like data.

def logbin(inarray, base=2.0, num_bin=30, bin_start=8):
    '''
    Bins values along a logarithmic x-axis at regular intervals. 

    Parameters:
    -----------
    inarray: Numpy Array (1 dimensional) to be binned.
    base: Float base of Logarithmic data. Defaults base 2.
    num_bin: Number of binning intervals. Defaults 30.
    bin_start: Index of the data to start binning. Binning low indices on log doesn't make 
    much sense. Any data below the start index is copied to the return array.

    Returns:
    -----------
    ret: Binned array of same dimension.
    '''

    min_log = get_max_log(bin_start, base) # Minimum log power depending on bin_start
    max_log = get_max_log(inarray.size, base) # Highest log power of the data
    # E.g. if data is 3000, highest of base two would be 2048, so return 11.
    # If data is 1000, highest of base two would be 512, so return 9. 
    # Anydata beyond that is averaged as one trailing bin.

    # Binning intervals courtesy of numpy.
    interval = np.logspace(min_log, max_log, num=num_bin, base=base, dtype='int')
    
    ret = np.zeros(inarray.size)
    ret[:bin_start] = inarray[:bin_start] # Data below bin_start is copied over.

    # Set indices in a bin to the average value of that bin.
    for i in range(len(interval)-1):
        if interval[i] != interval[i+1]:
            ret[interval[i]:interval[i+1]] = np.average(inarray[interval[i]:interval[i+1]])
        else:
            ret[interval[i]] = inarray[interval[i]]

    ret[2**max_log:] = np.average(inarray[2**max_log:]) # Trailing bin.
    return ret





### Helper Functions ###

def get_max_log(size, base):
    '''
    Given the size of an array, return the exponent of the maximum power of *base* that
    is smaller than the size. 

    E.g. if size is 3000 and base is 2, return 11, since 2^11 < 3000 < 2^12
    '''
    return (int(np.log2(size)) if base == 2.0 else int(np.log10(size)))


