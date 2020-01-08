'''
Simplified Logistic Regression Classifier

This function is very simplified in that I only use one predictor and an intercept for the optimization. I may expand on this later 
however my goal was mainly to review the concepts of this problem which are much the same as when having multiple predictors. 

The cost function is the residual deviance equation or -2 times the log likelihood.

Specifically this works by modelling the equation a + b*x_k = log(pi_k/(1-pi_k)) or the log odds (note pi_k represents the probability of success)
for trial k. More specifically pi_k = probability trial k is a success. 

I use gradient descent to then minimize the residual deviance. I've also added a finite differences gradient estimation that works well and is 
fun to experiment with. 

If you want to classify something as a 1/yes or a 0/no, you can use the disp_probabilities function with your desired x_values and simply 
test for if the probability is > .5 or whatever your cutoff is. 

'''


import numpy as np 
import pandas as pd 


class SimpleLogistic:

    def __init__(self, Xtrain, Ytrain):

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

        self.alpha = 0.1
        self.beta = 0.1

    def deviance(self):

        left = np.sum(self.Ytrain*self.alpha + self.beta*self.Xtrain*self.Ytrain)
        right = np.sum(np.log(1 + np.exp(self.alpha + self.beta*self.Xtrain)))

        log_likelihood = (left - right)

        return -2*log_likelihood

    def gradient(self):
        p1_alpha = -2*self.Ytrain
        p2_alpha = 2*np.exp(self.alpha + self.beta*self.Xtrain)
        p3_alpha = (1+np.exp(self.alpha + self.beta*self.Xtrain))

        partial_alpha = np.sum(p1_alpha + p2_alpha/p3_alpha)
        
        p2_beta = 2*self.Xtrain*np.exp(self.alpha + self.beta*self.Xtrain)
        p3_beta = (1+np.exp(self.alpha + self.beta*self.Xtrain))

        partial_beta = np.sum(p1_beta + p2_beta/p3_beta)

        return partial_alpha, partial_beta

    def minimize_deviance(self, learning_rate, max_iterations, verbose = False, tol = 1e-6):
        self.alpha = 0
        self.beta = 0

        deviance = self.deviance()

        cur_iter = 0
        norm_grad = np.linalg.norm(self.gradient(), np.inf)

        while(cur_iter < max_iterations and norm_grad > tol):
            partial_alpha, partial_beta = self.gradient()

            self.alpha = self.alpha - learning_rate*partial_alpha
            self.beta = self.beta - learning_rate*partial_beta

            deviance = self.deviance()

            if verbose and cur_iter%100 == 0:
                print("deviance || partial_alpha || partial_beta || norm_grad")
                print("%3f      %3f      %3f     %3f" % (deviance, partial_alpha, partial_beta, norm_grad))
            norm_grad = np.linalg.norm(self.gradient(), np.inf)
            cur_iter += 1

        print('\n Final Estimates:')
        print('Alpha = %4f'%(self.alpha))
        print('Beta = %4f'%(self.beta))
        print('Residual Deviance = ', deviance)
        print('Norm of gradient = ', norm_grad)
        return deviance

    def finite_dif_gradient(self):
        orig_a = self.alpha
        orig_b = self.beta

        deviance = self.deviance()

        self.alpha = self.alpha + 1e-8
        ndev = self.deviance()
        pA = (ndev - deviance)/1e-8

        self.alpha = orig_a
        self.beta += 1e-8

        ndev = self.deviance()
        pB = (ndev-deviance)/1e-8

        return pA, pB

    def disp_probabilities(self, xvalues):

        probs = []
        for val in xvalues:
            prob = np.exp(self.alpha + self.beta*val)/(1 + np.exp(self.alpha + self.beta*val))
            probs.append(prob)

        return probs




# X = np.array([2, 5, 10, 20, 25, 30, 2, 5, 10, 20, 25, 30])
# Y = np.array(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

# counts = np.array([428, 397, 330, 204, 94, 51])
# counts = np.concatenate(counts, 500-counts)
# Xtrain = np.repeat(X)

















