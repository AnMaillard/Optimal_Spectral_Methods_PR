import numpy as np
import pickle
import multiprocessing as mp
from Class import Spectral

"""
This runs the TAP/LAMP/MM spectral methods for complex Gaussian sensing matrices
"""
def run(alpha, m, channel, parameters, ensemble, N_average, return_eigenvalues, shared_matrix, lamp_closest_one, seed, verbosity):
    print("Starting alpha =", alpha)
    n = int(m/alpha)
    iterator = Spectral.Spectral(n_=n, alpha_=alpha, signal_="random", channel_=channel, parameters_=parameters, ensemble_=ensemble, N_average_=N_average, return_eigenvalues_= return_eigenvalues, shared_matrix_= shared_matrix, lamp_closest_one_ = lamp_closest_one, seed_= seed, verbosity_= verbosity)
    output = iterator.main() #The run function
    filename = "Data/tmp/results_"+ensemble+"_channel_"+channel+"_alpha_"+str(alpha)+"_m_"+str(m)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()
    print("Ending alpha = ", alpha)
    return output


#Lots of points
alphas = np.linspace(0.5, 4.0, 50)
#We fix m and change the value of n as alpha varies
m_list = 10000*np.ones_like(alphas)
#For small alpha we reduce a bit n to avoir too large values of n
for i,alpha in enumerate(alphas):
    if alpha <= 0.9:
        m_list[i] = 5000

ensemble = "gaussian_complex"
channel = "noiseless" #can be noiseless,poisson
parameters=[] #No parameters here
number_averages = 10
N_average = {"TAP":number_averages, "MM":number_averages, "LAMP":number_averages}
seed = False
verbosity = 2
return_eigenvalues = False #We do not save the spectrums
shared_matrix = True #Share the matrix among all methods. 
lamp_closest_one = True #Do we compute the LAMP closest to 1 eigenvalue (not relevant if return_eigenvalues==True)

print("Starting ", ensemble, " matrices.")
pool = mp.Pool(processes=12) #The mp pool
results = [pool.apply(run, args=(alpha, m_list[i], channel, parameters, ensemble, N_average, return_eigenvalues, shared_matrix, lamp_closest_one, seed, verbosity,)) for (i,alpha) in enumerate(alphas)]
#Save results
filename = "Data/results_"+ensemble+"_channel_"+channel+".pkl"
output = {'alphas':alphas,'results':results}
outfile = open(filename,'wb')
pickle.dump(output,outfile)
outfile.close()