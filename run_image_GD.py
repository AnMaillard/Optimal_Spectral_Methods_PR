import numpy as np
import pickle
import multiprocessing as mp
import matplotlib.image as mpimg
from skimage.transform import resize
from Class import Spectral

"""
This runs the TAP/LAMP spectral methods for a real image, averaging over the 3 RGB channels.
We do it for the matrix ensembles:
- The Haar measure on U(N)
- Partial dft matrices
- A product of complex Gaussian matrices with gamma = 1

This script also runs the GD after the spectral initialization, and saves its output (i.e. the estimated image)
"""

def run(RGB_channel, alpha, n, channel, parameters, ensemble, N_average, return_eigenvalues, shared_matrix, lamp_closest_one, seed, verbosity):
    print("Starting alpha =", alpha)
    iterator = Spectral.Spectral_with_GD(n_=n, alpha_=alpha, signal_="image", channel_=channel, parameters_=parameters, ensemble_=ensemble, N_average_=N_average, return_eigenvalues_= return_eigenvalues, shared_matrix_= shared_matrix, lamp_closest_one_ = lamp_closest_one, seed_= seed, verbosity_= verbosity)
    output = iterator.main() #The run function
    filename = "Data/tmp/results_real_image_"+ensemble+"_rgb_"+str(RGB_channel)+"_alpha_"+str(alpha)+"_with_GD.pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()
    print("Ending alpha = ", alpha)
    return output

ensembles = ["partial_dft","gaussian_product_complex_1"]
channel = "noiseless"
full_image = mpimg.imread('Data/image_1280_820.jpg')
RGB = full_image.shape[-1] #There are 3 RGB channels
for RGB_channel in range(RGB):
    print("This is the channel number",RGB_channel+1, "over the 3 RGB channels.")
    number_averages = 3
    N_average = {"TAP":number_averages, "MM":0, "LAMP":number_averages}
    seed = False
    verbosity = 2
    return_eigenvalues = False #We do not save the spectrums
    shared_matrix = True #Share the matrix among all methods. 
    lamp_closest_one = False #Do we compute the LAMP closest to 1 eigenvalue (not relevant if return_eigenvalues==True)

    for ensemble in ensembles:
        print("Starting ", ensemble, " matrices.")
        reduction_factor = 10
        alphas = np.linspace(4.,1.5,25)
        if  ensemble == "gaussian_product_complex_1": #Then we need different alphas, so we can also take a bigger image
            alphas = np.linspace(2.,0.25,25)
            reduction_factor = 10
        #We reduce the size of the image
        data = resize(full_image,(full_image.shape[0] // reduction_factor, full_image.shape[1] // reduction_factor), anti_aliasing=True)
        n = data.shape[0]*data.shape[1]
        #We remove the mean and normalize the data to have variance n
        mean, std = np.mean(data,axis=(0,1)), np.std(data,axis=(0,1))
        data = (data - mean)/std #This way it has squared norm n
        parameters={"Xstar":data[:,:,RGB_channel].flatten(),"learning_rate_initial":0.1,"epsilon":1e-4,"MAX_ITER":1000}

        pool = mp.Pool(processes=12) #The mp pool
        results = [pool.apply(run, args=(RGB_channel,alpha, n, channel, parameters, ensemble, N_average, return_eigenvalues, shared_matrix, lamp_closest_one, seed, verbosity,)) for (i,alpha) in enumerate(alphas)]
        #Save results
        filename = "Data/results_real_image_"+ensemble+"_rgb_"+str(RGB_channel)+"_with_GD.pkl"
        output = {'alphas':alphas,'results':results}
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()