import numpy as np
from scipy.linalg import sqrtm

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
#import tensorflow_gan as tfgan

def calculate_statistics(features):
        mean = np.mean(features, axis=0)
        covariance = np.cov(features, rowvar=False)
        return mean, covariance

class FID():
    def __init__(self, inception_model, output_tensor = 'pool_3:0', input_shape=(128, 128, 3) ):
        if inception_model is not None:
            self.inception_model = inception_model
        else:
            self.inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
        self.output_tensor = output_tensor
        self.input_shape = input_shape
        
    def extract_features(self, images, expand = False):
        # Resize images to 299x299 (required for InceptionV3)
        images_resized = tf.image.resize(images, (128,128))
        # Preprocess the images
        preprocessed_images = preprocess_input(images_resized.numpy())
        # Extract features using InceptionV3
        if expand:
            preprocessed_images = np.expand_dims(preprocessed_images, axis=0)
        features = self.inception_model(preprocessed_images, training=False)
        return features
    
    def get_real_features(self, real_images):
        
        real_features = self.extract_features(real_images, expand=True)
        
        self.mu_real, self.sigma_real = calculate_statistics(real_features)
        return self.mu_real, self.sigma_real
    
    def get_generated_features(self, fake_images):
        
        generated_features = self.extract_features(fake_images)
        
        self.mu_generated, self.sigma_generated = calculate_statistics(generated_features)
        return self.mu_generated, self.sigma_generated
    
    def calculate_fid(self, mu_real, sigma_real, mu_gen, sigma_gen):
        if mu_real not in locals():
            mu_real = self.mu_real
        if sigma_real not in locals():
            sigma_real = self.sigma_real
        if mu_gen not in locals():
            mu_gen = self.mu_generated
        if sigma_gen not in locals():
            sigma_gen = self.sigma_generated
        
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