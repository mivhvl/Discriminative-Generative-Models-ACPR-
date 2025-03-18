import numpy as np
from scipy.linalg import sqrtm

import tensorflow as tf
import tensorflow_gan as tfgan

def calculate_statistics(features):
        mean = np.mean(features, axis=0)
        covariance = np.cov(features, rowvar=False)
        return mean, covariance

class FID():
    def __init__(self, inception_model = tfgan.eval.run_inception , output_tensor = 'pool_3:0' ):
        self.inception_model = inception_model
        self.output_tensor = output_tensor
    
    def get_real_features(self, real_images):
        real_features = self.inception_model(real_images, output_tensor= self.output_tensor)
        self.mu_real, self.sigma_real = calculate_statistics(real_features)
        return self.mu_real, self.sigma_real
    
    def get_generated_features(self, fake_images):
        generated_features = self.inception_model(fake_images, output_tensor= self.output_tensor)
        self.mu_generated, self.sigma_generated = calculate_statistics(generated_features)
        return self.mu_generated, self.sigma_generated
    
    def calculate_fid(self, mu_real, sigma_real, mu_gen, sigma_gen):
        if mu_real not in locals():
            mu_real = self.mu_real
        if sigma_real not in locals():
            sigma_real = self.sigma_real
        if mu_gen not in locals():
            mu_gen = self.mu_gen
        if sigma_gen not in locals():
            sigma_gen = self.sigma_gen
        
        # Calculate the difference between means
        diff = mu_real - mu_gen 
        
        # Calculate the square root of the product of covariance matrices
        covmean = sqrtm(sigma_real @ sigma_gen) 
        # Handle possible numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        # Fréchet Distance formula
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid


# - ----------------------------------------------------------------------------------------------------------------------------------------
# calculate frechet inception distance 
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid