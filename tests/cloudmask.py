import scipy.signal
import numpy as np

def create_mask(start,end,session):
    s2_sceneclassification = session.imagecollection("S2_FAPAR_SCENECLASSIFICATION_V102_PYRAMID",
                                                     bands=["classification"])

    classification = s2_sceneclassification.band('classification')

    def makekernel(iwindowsize):
        kernel_vect = scipy.signal.windows.gaussian(iwindowsize, std=iwindowsize / 3.0, sym=True)
        kernel = np.outer(kernel_vect, kernel_vect)
        kernel = kernel / kernel.sum()
        return kernel

    #in openEO, 1 means mask (remove pixel) 0 means keep pixel

    #keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17))
    #remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    first_mask = first_mask > 0.057

    #remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10)
    second_mask = second_mask.apply_kernel(makekernel(161))
    second_mask = second_mask > 0.1

    #TODO: the use of filter_temporal is a trick to make cube merging work, needs to be fixed in openeo client
    return first_mask.filter_temporal(start,end) | second_mask.filter_temporal(start,end)

