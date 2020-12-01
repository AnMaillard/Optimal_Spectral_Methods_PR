#Class implementing the spectral methods for phase retrieval problem

import time, math, random
import numpy as np 
from scipy import linalg
from scipy.fftpack import dct

def power_method(n, matrix, MAX_ITERATIONS = 1000, epsilon = 1e-4):
    #Does a power method on the matrix of size n
    x = np.random.normal(0, 1., n) #Initial point
    x *= np.sqrt(n) / linalg.norm(x) #Normalize
    counter, converged = 0, False
    while (not(converged) and counter < MAX_ITERATIONS):
        z = np.dot(matrix, x) 
        z *= np.sqrt(n) / linalg.norm(z) #Normalize
        is_eigenvector = np.abs((z/z[0])*(x[0]/x) - 1.) #If x is an eigenvector, then because of the normalization z = Exp[i \theta] * x and thus this vector should be all 0s
        converged = (np.amax(is_eigenvector) <= epsilon)
        x = z
        counter += 1
    return converged, x

#The class of the iterator, without Gradient Descent
class Spectral:

    def __init__(self, n_, alpha_, signal_, channel_, parameters_, ensemble_, N_average_, return_eigenvalues_, shared_matrix_, lamp_closest_one_, seed_ = False, verbosity_ = 0):
        self.verbosity = verbosity_
        self.return_eigenvalues = return_eigenvalues_ #Do we return the full spectra of the methods
        self.shared_matrix = shared_matrix_ #Do the different instances of the methods share the same instance of the sensing matrix
        self.n = int(n_)
        self.alpha = alpha_
        self.m = int(alpha_ * n_)
        self.signal = signal_
        assert self.signal in ["random","image"], "ERROR: The signal must be either of type random or image"
        
        self.channel = channel_
        self.ensemble = ensemble_

        self.type_variable = "real"
        if self.ensemble in ["unitary","gaussian_complex","partial_dft"] or self.ensemble[:25] == "gaussian_product_complex_":
            self.type_variable = "complex"
        if seed_:
            self.seed = 1
        else:
            self.seed = time.time_ns() % 1000000
        np.random.seed(self.seed)
        self.parameters = parameters_
        self.lamp_closest_one = lamp_closest_one_ #Do we compute the eigenvalue of LAMP closest to 1

        if self.channel == "poisson":
            self.snr0 = self.parameters['snr0']
            self.snr = self.parameters['snr']
            if self.snr != self.snr0:
                print("Non Bayes optimal Poisson, lambda_0 =", self.snr0, " and lambda =", self.snr)

        self.N_average = N_average_ #Number of averages. This is a dictionary
        #Initializing the overlaps and eigenvalues
        if self.N_average["LAMP"] > 0:
            self.q_LAMP = {"Largest":[], "Closest to 1":[]}
            self.ev_LAMP = {"Largest":[],"Closest to 1":[]}

        if self.N_average["TAP"] > 0:
            self.q_TAP = {"Largest":[]}
            self.ev_TAP = {"Largest":[]}

        if self.N_average["MM"] > 0:
            self.q_MM = {"Largest":[]}
            self.ev_MM = {"Largest":[]}
            
        #The Spectra
        if self.return_eigenvalues:
            self.spectrum = {"LAMP":[],"TAP":[],"MM":[]}

        if self.shared_matrix:
            N_average_max = max(self.N_average["LAMP"],self.N_average["MM"],self.N_average["TAP"])
            self.N_average_shared = N_average_max
            for method in ["LAMP","MM","TAP"]:
                assert self.N_average[method] in [0,N_average_max], "ERROR : Can not share the matrix if the number of averages are different !" 
        
        if self.signal == "image":
            #For a real image, we also output the estimated vector xhat (normalized with squared norm equal to n)
            if self.N_average["LAMP"] > 0:
                self.xhat_LAMP = {"Largest":[], "Closest to 1":[]}
            if self.N_average["TAP"] > 0:
                self.xhat_TAP = {"Largest":[]}
            if self.N_average["MM"] > 0:
                self.xhat_MM = {"Largest":[]}
        
    def generate_matrix(self):
        #Generates a matrix with the given ensemble
        #We normalize phi such that (1/(n*m))*sum(phi_{\mu i}^2) = 1 in the large n,m limit

        if self.ensemble == "gaussian":
            self.phi = np.random.normal(0, 1., (self.m, self.n))

        elif self.ensemble == "gaussian_complex":
            self.phi = (1./np.sqrt(2.))*(np.random.normal(0, 1., (self.m, self.n)) + 1j*np.random.normal(0, 1., (self.m, self.n)))
        
        elif self.ensemble == "orthogonal":
            assert self.m >= self.n, "ERROR : For an orthogonal matrix, we need alpha >= 1 !"
            #We first generate a m x m orthogonal matrix by taking the QR decomposition of a random m x m gaussian matrix
            gaussian_matrix = np.random.normal(0, 1., (self.m,self.m))
            O, R = linalg.qr(gaussian_matrix)
            #Then we multiply on the right by the signs of the diagonal of R to draw it from the Haar measure, as shown in
            #Mezzadri, F. "How to generate random matrices from the classical compact groups." arXiv preprint math-ph/0609050
            D = np.diagonal(R)
            Lambda = D / np.abs(D)
            O = np.multiply(O, Lambda)
            self.phi = np.sqrt(self.m) * O[:, 0:self.n]#Then we take its n first columns, correctly normalized

        elif self.ensemble == "hadamard":
            assert self.m >= self.n, "ERROR : For an Hadamard matrix, we need alpha >= 1 !"
            #We first generate a m x m deterministic Hadamard matrix
            assert math.log2(self.m).is_integer(),  "ERROR : For an Hadamard matrix, we need m to be a power of 2"
            hadamard_matrix = linalg.hadamard(self.m)
            #Then we take randomly n columns of it 
            columns_list = random.sample(range(self.m), self.n)
            DSmatrix = hadamard_matrix[:,columns_list]
            #Then we multiply on the right by a diagonal of random +- 1
            random_signs = (2*np.random.randint(0,2,size=(self.n))-1)
            O = np.multiply(DSmatrix,random_signs)
            normalization = np.sum(np.square(O))/(self.m*self.n) #Normalization
            self.phi = O / np.sqrt(normalization)

        elif self.ensemble == "dct":
            assert self.m >= self.n, "ERROR : For a DCT matrix, we need alpha >= 1 !"
            #We first generate a DCT matrix of size m
            dct_matrix = np.transpose(dct(np.eye(self.m), norm = 'ortho'))
            #Then we take randomly n columns of it 
            columns_list = random.sample(range(self.m), self.n)
            DSmatrix = dct_matrix[:,columns_list]
            #Then we multiply on the right by a diagonal of random +- 1
            random_signs = (2*np.random.randint(0,2,size=(self.n))-1)
            D = np.multiply(DSmatrix,random_signs)
            normalization = np.sum(np.square(D))/(self.m*self.n) #Normalization
            self.phi = D / np.sqrt(normalization)

        elif self.ensemble == "unitary":
            assert self.m >= self.n, "ERROR : For an unitary matrix, we need alpha >= 1 !"
            #We first generate a m x m unitary matrix by taking the QR decomposition of a random m x m complex gaussian matrix
            gaussian_matrix = (1./np.sqrt(2.))*(np.random.normal(0, 1., (self.m, self.m)) + 1j*np.random.normal(0, 1., (self.m, self.m)))
            U, R = linalg.qr(gaussian_matrix)
            #Then we multiply on the right by the phases of the diagonal of R to draw it from the Haar measure, as shown in
            #Mezzadri, F. "How to generate random matrices from the classical compact groups." arXiv preprint math-ph/0609050
            D = np.diagonal(R)
            Lambda = D / np.abs(D)
            U = np.multiply(U, Lambda)
            #Then we take its n first columns, correctly normalized
            self.phi = np.sqrt(self.m) * U[:, 0:self.n]

        elif self.ensemble == "partial_dft":
            assert self.m >= self.n, "ERROR : For an unitary matrix, we need alpha >= 1 !"
            #We first generate a DFT matrix of size m
            dft_matrix = linalg.dft(self.m)
            #Then we take randomly n columns of it 
            columns_list = random.sample(range(self.m), self.n)
            DSmatrix = dft_matrix[:,columns_list]
            phases = np.ones(self.n)
            #Then we multiply on the right by a diagonal of random phases
            phases = np.exp(1j*np.random.uniform(0,2*np.pi,self.n))
            PDFT_matrix = np.multiply(DSmatrix,phases)
            normalization = np.sum(np.square(np.abs(PDFT_matrix)))/(self.m*self.n)
            self.phi = PDFT_matrix / np.sqrt(normalization)

        elif self.ensemble[0:17] == "gaussian_product_":
            #We extract the beta = k/n parameter from the name
            if self.type_variable == "real":
                beta = float(self.ensemble[17:])
            elif self.type_variable == "complex":
                beta = float(self.ensemble[25:])
            k = int(beta*self.n)
            if self.type_variable == "real":
                self.phi = (1./np.sqrt(k)) * np.random.normal(0, 1., (self.m, k)) @ np.random.normal(0, 1., (k, self.n))
            elif self.type_variable == "complex":
                W1 = np.random.normal(0, 1., (self.m, k)) + 1j *np.random.normal(0, 1., (self.m, k))
                W2 = np.random.normal(0, 1., (k, self.n)) + 1j *np.random.normal(0, 1., (k, self.n))
                self.phi = (1./(2*np.sqrt(k))) * W1 @ W2

        else:
            assert False, "ERROR : Not implemented ensemble"

    def initialize(self):
        #We generate the data, and compute the dgout
        self.Z, self.Y = np.zeros(self.m), np.zeros(self.m)

        #Generate Xstar
        if self.signal == "random":
            if self.type_variable == "real":
                self.Xstar = np.random.normal(0., 1., self.n)
                self.Xstar *= np.sqrt(self.n)/np.linalg.norm(self.Xstar)
            elif self.type_variable == "complex":
                self.Xstar = np.random.normal(0., 1., self.n) + 1j*np.random.normal(0., 1., self.n) 
                self.Xstar *= np.sqrt(self.n)/np.linalg.norm(self.Xstar)
            else:
                assert False, "ERROR : Unknown variable type"
        elif self.signal == "image": #For a real image the signal is given in the parameters
            self.Xstar = self.parameters["Xstar"]

        self.generate_matrix()

        #Generate the Y vector
        if self.channel == "noiseless":
            self.Y = np.square(np.abs((1./np.sqrt(self.n)) * np.dot(self.phi, self.Xstar)))
        elif self.channel == "poisson":
            s2 = np.square(np.abs((1./np.sqrt(self.n)) * np.dot(self.phi, self.Xstar)))
            #We draw Y from the Poisson distribution with parameter lambda_0*s^2
            self.Y = np.random.poisson(self.snr0*s2,size = (self.m,))
        else:
            assert False, "ERROR : Not implemented channel :"+self.channel
        
        #Compute Z, used in the spectral methods
        if self.channel == "noiseless":
            self.Z = self.Y - 1.
        elif self.channel == "poisson":
            if self.type_variable == "real":
                self.Z = 2*(self.Y-self.snr)/(1+2*self.snr) #Since the variance is one, this in the real case
            elif self.type_variable == "complex":
                self.Z = (self.Y-self.snr)/(1+self.snr) #Since the variance is one, this is the complex cas
            else:
                assert "ERROR :  Unknown variable type"
        else:
            assert False, "ERROR : Not implemented channel :"+self.channel

    def run_LAMP(self, save_xhat = False):
        LAMP =  np.multiply(self.phi @ np.conjugate(np.transpose(self.phi))/self.n - np.eye(self.m), self.Z)

        if self.return_eigenvalues:
            if self.verbosity >= 2:
                print("Computing the full LAMP eigenvalue decomposition...")
            evalues_LAMP, evectors_LAMP = linalg.eig(LAMP)
            #Now we isolate the largest eigenvalue in real part and the eigenvalue closest to 1
            index_largest = np.argmax(np.real(evalues_LAMP)) #Largest in real part
            index_closest_one = np.argmin(np.abs(evalues_LAMP-1.)+3*np.abs(np.imag(evalues_LAMP))) #Closest to 1 and small imaginary part (we add a regularization term)
            self.ev_LAMP["Largest"].append(evalues_LAMP[index_largest])
            self.ev_LAMP["Closest to 1"].append(evalues_LAMP[index_closest_one])

            #Compute the estimated vectors
            transfer_matrix = np.multiply(np.conjugate(np.transpose(self.phi)),self.Z)/np.sqrt(self.n)
            xLargest = np.dot(transfer_matrix, evectors_LAMP[:,index_largest])
            xClosest = np.dot(transfer_matrix, evectors_LAMP[:,index_closest_one])

            #Now we renormalize
            xLargest *= np.sqrt(self.n)/linalg.norm(xLargest)
            xClosest *= np.sqrt(self.n)/linalg.norm(xClosest)

            #And now the overlaps
            self.q_LAMP["Largest"].append(np.abs(np.dot(np.conjugate(xLargest),self.Xstar)/self.n))
            self.q_LAMP["Closest to 1"].append(np.abs(np.dot(np.conjugate(xClosest),self.Xstar)/self.n))
        
            #The full estimators if needed
            if save_xhat:
                self.xhat_LAMP["Largest"].append(xLargest)
                self.xhat_LAMP["Closest to 1"].append(xClosest)

            #Add the full spectrum
            self.spectrum["LAMP"].append(evalues_LAMP)

        else: #Here we use power iterations for the largest eigenvalue, and inverse iterations for the closest to 1 if needed
            MAX_ITERATIONS, epsilon = 1000, 1e-4

            #Power method for the largest eigenvalue
            if self.verbosity >= 2:
                print("Computing the largest eigenvalue of LAMP with power method...")
            t0 = time.time()
            converged_largest, evector_largest = power_method(self.m, LAMP+10.*np.eye(self.m), MAX_ITERATIONS=MAX_ITERATIONS, epsilon=epsilon) #We shift the LAMP matrix to enhance the convergence of the iterations
            if not(converged_largest) and self.verbosity >= 2:
                 print("The power method on LAMP did not converge !!")
            t1 = time.time()
            if self.verbosity >= 2:
                print("Power method took",t1-t0,"seconds.")
            self.ev_LAMP["Largest"].append(np.dot(np.conjugate(evector_largest),np.dot(LAMP,evector_largest))/self.m)

            if self.lamp_closest_one:
                print("Computing the closest to 1 eigenvalue of LAMP with inverse method...")
                t0 = time.time()
                inverse_matrix = linalg.inv(LAMP - np.eye(self.m))
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Inverse matrix computed in",t1-t0,"seconds !")
                t0 = time.time()
                converged_closest, evector_closest = power_method(self.m, inverse_matrix, MAX_ITERATIONS=MAX_ITERATIONS, epsilon=epsilon)
                if not(converged_closest) and self.verbosity >= 2:
                    print("The inverse iterations on LAMP did not converge !!")
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Inverse method took",t1-t0,"seconds.")
                self.ev_LAMP["Closest to 1"].append(np.dot(np.conjugate(evector_closest),np.dot(LAMP,evector_closest))/self.m)

            #Compute the estimated vectors
            transfer_matrix = np.multiply(np.conjugate(np.transpose(self.phi)),self.Z)/np.sqrt(self.n)
            xLargest = np.dot(transfer_matrix, evector_largest)
            xLargest *= np.sqrt(self.n)/linalg.norm(xLargest)
            self.q_LAMP["Largest"].append(np.abs(np.dot(np.conjugate(xLargest),self.Xstar)/self.n))

            if self.lamp_closest_one:
                xClosest = np.dot(transfer_matrix, evector_closest)
                xClosest *= np.sqrt(self.n)/linalg.norm(xClosest)
                self.q_LAMP["Closest to 1"].append(np.abs(np.dot(np.conjugate(xClosest),self.Xstar)/self.n))
        
            #The full estimators if needed
            if save_xhat:
                self.xhat_LAMP["Largest"].append(xLargest)
                if self.lamp_closest_one:
                    self.xhat_LAMP["Closest to 1"].append(xClosest)

        if self.verbosity >= 2:
            print("Done !")

    def run_TAP(self, save_xhat = False):
        TAP = (1./self.n)*np.multiply(np.conjugate(np.transpose(self.phi)),self.Z/(1.+self.Z)) @ self.phi #This is M_TAP + (1/rho)

        if self.return_eigenvalues:
            if self.verbosity >= 2:
                print("Computing the full set of TAP eigenvalues...")
            evalues_TAP, evectors_TAP = linalg.eigh(TAP)
            self.ev_TAP["Largest"].append(evalues_TAP[self.n-1])
            xLargest = (np.sqrt(self.n)/linalg.norm(evectors_TAP[:,self.n-1]))*evectors_TAP[:,self.n-1]
            self.spectrum["TAP"].append(evalues_TAP)
        else: #Inverse iterations, as we know that the largest eigenvalue of TAP concentrates on 1
            MAX_ITERATIONS, epsilon = 1000, 1e-4
            if self.verbosity >= 2:
                print("Computing the first eigenvalue and eigenvector with inverse iteration...")
            t0 = time.time()
            inverse_matrix = linalg.inv(TAP - np.eye(self.n))
            t1 = time.time()
            if self.verbosity >= 2:
                print("Inverse matrix computed in",t1-t0,"seconds !")
            t0 = time.time()
            converged_largest, evector_largest = power_method(self.n, inverse_matrix, MAX_ITERATIONS=MAX_ITERATIONS, epsilon=epsilon)
            if not(converged_largest) and self.verbosity >= 2:
                 print("The inverse iteration on TAP did not converge !!")
            t1 = time.time()
            if self.verbosity >= 2:
                print("Inverse method took",t1-t0,"seconds.")
            self.ev_TAP["Largest"].append(np.dot(np.conjugate(evector_largest),np.dot(TAP,evector_largest))/self.n)
            xLargest = evector_largest

        #Adding now the overlaps
        self.q_TAP["Largest"].append(np.abs(np.dot(np.conjugate(xLargest),self.Xstar)/self.n))

        #The full estimators if needed
        if save_xhat:
            self.xhat_TAP["Largest"].append(xLargest)

        if self.verbosity >= 2:
            print("Done !")
        
    def run_MM(self, save_xhat = False):
        factor = 2 #This is 2/beta
        if self.type_variable == "complex":
            factor = 1
        MM = (1./self.n)*np.multiply(np.conjugate(np.transpose(self.phi)),self.Z/(np.sqrt(factor*self.alpha)+self.Z)) @ self.phi
        #Here (legacy method) we use eigh rather than power methods
        if not self.return_eigenvalues:
            if self.verbosity >= 2:
                print("Computing the first MM eigenvalues...")
            evalues_MM, evectors_MM = linalg.eigh(MM, eigvals = (self.n-2, self.n-1))
            if self.verbosity >= 2:
                print("Done !")
            self.ev_MM["Largest"].append(evalues_MM[1])
            xLargest = (np.sqrt(self.n)/linalg.norm(evectors_MM[:,1]))*evectors_MM[:,1]
        else:
            if self.verbosity >= 2:
                print("Computing the full set of MM eigenvalues...")
            evalues_MM, evectors_MM = linalg.eigh(MM)
            if self.verbosity >= 2:
                print("Done !")
            self.ev_MM["Largest"].append(evalues_MM[self.n-1])
            xLargest = (np.sqrt(self.n)/linalg.norm(evectors_MM[:,self.n-1]))*evectors_MM[:,self.n-1]
            self.spectrum["MM"].append(evalues_MM)

        self.q_MM["Largest"].append(np.abs(np.dot(np.conjugate(xLargest),self.Xstar)/self.n))

        #The full estimators if needed
        if save_xhat:
            self.xhat_MM["Largest"].append(xLargest)

    def prepare_output(self):
        self.output = {}
        if self.N_average["LAMP"] > 0:
            self.output["q_LAMP"] = self.q_LAMP
            self.output["ev_LAMP"] = self.ev_LAMP
        if self.N_average["TAP"] > 0:
            self.output["q_TAP"] = self.q_TAP
            self.output["ev_TAP"] = self.ev_TAP
        if self.N_average["MM"] > 0:
            self.output["q_MM"] = self.q_MM
            self.output["ev_MM"] = self.ev_MM

        if self.return_eigenvalues:
            self.output["spectrums"] = self.spectrum
        
        if self.signal == "image": #Then we save the estimators
            if self.N_average["LAMP"] > 0:
                self.output["xhat_LAMP"] = self.xhat_LAMP
            if self.N_average["TAP"] > 0:
                self.output["xhat_TAP"] = self.xhat_TAP
            if self.N_average["MM"] > 0:
                self.output["xhat_MM"] = self.xhat_MM
        self.output["seed"] = self.seed
    
    def run_not_shared(self):
        #Does all the iterations, without sharing the matrix instance between the methods
        if self.N_average["LAMP"] > 0:
            for i in range(self.N_average["LAMP"]):
                if self.verbosity >= 1:
                    print("LAMP Iteration", i+1,'/',self.N_average["LAMP"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_LAMP(save_xhat = (self.signal == "image"))
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of LAMP iterations")

        if self.N_average["TAP"] > 0:
            for i in range(self.N_average["TAP"]):
                if self.verbosity >= 1:
                    print("TAP Iteration", i+1,'/',self.N_average["TAP"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_TAP(save_xhat = (self.signal == "image"))
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of TAP iterations")

        if self.N_average["MM"] > 0:
            for i in range(self.N_average["MM"]):
                if self.verbosity >= 1:
                    print("MM Iteration", i+1,'/',self.N_average["MM"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_MM(save_xhat =(self.signal == "image"))
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of MM iterations")

        self.prepare_output()
        return self.output

    def run_shared(self):
        #Does all the iterations, sharing the matrix instance between the methods
        for i in range(self.N_average_shared):
            t0 = time.time()
            self.initialize()
            t1 = time.time()
            if self.verbosity >= 2:
                print("Initialization done, it took", round(t1 - t0,4), "seconds.")

            if self.N_average["LAMP"] > 0:
                if self.verbosity >= 1:
                    print("Starting LAMP Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_LAMP(save_xhat = (self.signal=="image"))
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ", round(t1-t0,4), " seconds.")

            if self.N_average["TAP"] > 0:
                if self.verbosity >= 1:
                    print("Starting TAP Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_TAP(save_xhat = (self.signal == "image"))
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ", round(t1-t0,4), " seconds.")

            if self.N_average["MM"] > 0:
                if self.verbosity >= 1:
                    print("Starting MM Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_MM(save_xhat =(self.signal == "image"))
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ", round(t1-t0,4), " seconds.")
            
        self.prepare_output()
        return self.output

    def main(self):
        if self.shared_matrix:
            return self.run_shared()
        else:
            return self.run_not_shared()

#A subclass, including gradient descent methods
class Spectral_with_GD(Spectral):
    def __init__(self, n_, alpha_, signal_, channel_, parameters_, ensemble_, N_average_, return_eigenvalues_, shared_matrix_, lamp_closest_one_, seed_ = False, verbosity_ = 0):
        Spectral.__init__(self, n_, alpha_, signal_, channel_, parameters_, ensemble_, N_average_, return_eigenvalues_, shared_matrix_, lamp_closest_one_, seed_, verbosity_)
        self.lr = parameters_["learning_rate_initial"] 
        self.epsilon = parameters_["epsilon"]
        self.MAX_ITER = parameters_["MAX_ITER"]
        self.x, self.x_old = np.zeros(self.n), np.zeros(self.n)
        self.current_gradient, self.old_gradient = np.zeros(self.n), np.zeros(self.n)

        #We also output the estimated vector xhat (normalized with squared norm equal to n). This can be useless if the signal is a real image as this was already done in the parent class Spectral
        if self.N_average["LAMP"] > 0:
            self.xhat_LAMP = {"Largest":[], "Closest to 1":[]}
        if self.N_average["TAP"] > 0:
            self.xhat_TAP = {"Largest":[]}
        if self.N_average["MM"] > 0:
            self.xhat_MM = {"Largest":[]}

        self.xhat_LAMP_GD, self.q_LAMP_GD = [], []
        self.xhat_TAP_GD, self.q_TAP_GD = [], []
        self.xhat_MM_GD, self.q_MM_GD = [], []
        assert self.channel == "noiseless", "ERRROR: Only noiseless channel supported for GD for now"
    
    def loss_value(self, x):
        projection = self.phi @ x / np.sqrt(self.n)
        return (1./2)*np.sum((self.Y - np.abs(projection)**2)**2)/self.m
    
    def gradient(self,x):
        projection = np.dot(self.phi,x) / np.sqrt(self.n)
        gradient = np.dot((np.abs(projection)**2 - self.Y)*projection,np.conjugate(self.phi)/np.sqrt(self.n))
        return gradient
    
    def iterate_GD(self):
        self.current_gradient = self.gradient(self.x)
        #We update the learning rate with the Barzilai–Borwein method
        denominator = linalg.norm(self.current_gradient - self.old_gradient)**2
        numerator = np.real(np.dot(np.conjugate(self.x-self.x_old),self.current_gradient - self.old_gradient))
        self.lr = np.abs(numerator)/denominator
        x_new = self.x - self.lr * self.current_gradient
        x_new *= np.sqrt(self.n)/linalg.norm(x_new)
        #Compute the total variation
        variation = linalg.norm(self.x - x_new)/np.sqrt(self.n) #Norm difference, for random vectors it is of order 1
        #Update the variables
        self.x_old = self.x
        self.x = x_new

        self.old_x = self.x
        self.old_gradient = self.current_gradient

        return (variation <= self.epsilon)
    
    def run_GD(self,X0 = None):
        """
        Returns the GD with initial vector X0 (X0 is actually a list for all the different iterations)
        If X0 = None, we use a random initialization
        """
        if X0 is None:
            X0 = np.random.normal(0., 1., self.n)
            X0 *= np.sqrt(self.n)/np.linalg.norm(X0)
        self.x = X0 #The variable self.x is used as the current estimator (not very clean)
        q_original = np.abs(np.dot(np.conjugate(self.x),self.Xstar)/self.n)
        converged = False
        k = 0 
        while not(converged) and k < self.MAX_ITER:
            converged = self.iterate_GD()
            k += 1
        q = np.abs(np.dot(np.conjugate(self.x),self.Xstar)/self.n)

        if self.verbosity >= 1:
            print("Convergence of GD:",converged, "and original/final overlaps:",q_original,"/",q)
        return self.x, q

    #Overwrite of the parent class function
    def run_shared(self):
        for i in range(self.N_average_shared):
            t0 = time.time()
            self.initialize()
            t1 = time.time()
            if self.verbosity >= 2:
                print("Initialization done, it took", round(t1 - t0,4), "seconds.")

            if self.N_average["LAMP"] > 0:
                if self.verbosity >= 1:
                    print("Starting LAMP Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_LAMP(save_xhat=True)
                xhat, q = self.run_GD(X0 = self.xhat_LAMP["Largest"][i])
                self.xhat_LAMP_GD.append(xhat)
                self.q_LAMP_GD.append(q)
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time =", round(t1-t0,4), "seconds.")

            if self.N_average["TAP"] > 0:
                if self.verbosity >= 1:
                    print("Starting TAP Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_TAP(save_xhat= True)
                xhat, q = self.run_GD(X0 = self.xhat_TAP["Largest"][i])
                self.xhat_TAP_GD.append(xhat)
                self.q_TAP_GD.append(q)
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time =", round(t1-t0,4), "seconds.")
            
            if self.N_average["MM"] > 0:
                if self.verbosity >= 1:
                    print("Starting MM Iteration", i+1,'/',self.N_average_shared)
                t0 = time.time()
                self.run_MM(save_xhat=True)
                xhat, q = self.run_GD(X0 = self.xhat_MM["Largest"][i])
                self.xhat_MM_GD.append(xhat)
                self.q_MM_GD.append(q)
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Time =", round(t1-t0,4), "seconds.")

        self.prepare_output()
        return self.output

    #Overwrite of the parent class function
    def run_not_shared(self):
        if self.N_average["LAMP"] > 0:
            for i in range(self.N_average["LAMP"]):
                if self.verbosity >= 1:
                    print("LAMP Iteration", i+1,'/',self.N_average["LAMP"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_LAMP(save_xhat = True)
                xhat, q = self.run_GD(X0 = self.xhat_LAMP["Largest"][i])
                self.xhat_LAMP_GD.append(xhat)
                self.q_LAMP_GD.append(q)
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of LAMP iterations")

        if self.N_average["TAP"] > 0:
            for i in range(self.N_average["TAP"]):
                if self.verbosity >= 1:
                    print("TAP Iteration", i+1,'/',self.N_average["TAP"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_TAP(save_xhat = True)
                xhat, q = self.run_GD(X0 = self.xhat_TAP["Largest"][i])
                self.xhat_TAP_GD.append(xhat)
                self.q_TAP_GD.append(q)
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of TAP iterations")

        if self.N_average["MM"] > 0:
            for i in range(self.N_average["MM"]):
                if self.verbosity >= 1:
                    print("MM Iteration", i+1,'/',self.N_average["MM"])
                t0 = time.time()
                self.initialize()
                t1 = time.time()
                if self.verbosity >= 2:
                    print("Initialization done, it took", t1 - t0, "seconds.")
                time_0 = time.time()
                self.run_MM(save_xhat =True)
                xhat, q = self.run_GD(X0 = self.xhat_MM["Largest"][i])
                self.xhat_MM_GD.append(xhat)
                self.q_MM_GD.append(q)
                time_1 = time.time()
                if self.verbosity >= 2:
                    print("Time = ",time_1-time_0, " seconds.")
            print("End of MM iterations")

        self.prepare_output()
        return self.output

    #Overwrite of the parent class function
    def prepare_output(self):
        self.output = {}
        if self.N_average["LAMP"] > 0:
            self.output["q_LAMP"] = self.q_LAMP
            self.output["ev_LAMP"] = self.ev_LAMP
            self.output["xhat_LAMP"] = self.xhat_LAMP
            self.output["xhat_LAMP_GD"] = self.xhat_LAMP_GD
            self.output["q_LAMP_GD"] = self.q_LAMP_GD
        if self.N_average["TAP"] > 0:
            self.output["q_TAP"] = self.q_TAP
            self.output["ev_TAP"] = self.ev_TAP
            self.output["xhat_TAP"] = self.xhat_TAP
            self.output["xhat_TAP_GD"] = self.xhat_TAP_GD
            self.output["q_TAP_GD"] = self.q_TAP_GD
        if self.N_average["MM"] > 0:
            self.output["q_MM"] = self.q_MM
            self.output["ev_MM"] = self.ev_MM
            self.output["xhat_MM"] = self.xhat_MM
            self.output["xhat_MM_GD"] = self.xhat_MM_GD
            self.output["q_MM_GD"] = self.q_MM_GD

        if self.return_eigenvalues:
            self.output["spectrums"] = self.spectrum
        self.output["seed"] = self.seed