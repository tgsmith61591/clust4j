package com.clust4j.kernel;

/**
 * The exponential kernel is closely related to the {@link GaussianKernel}, 
 * with only the square of the norm left out. It is also a radial basis 
 * function kernel.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class ExponentialKernel extends LaplacianKernel {
	public static final double DEFAULT_EXPONENTIAL	= 1;
	public static final double DEFAULT_SIGMA_EXP	= 2;
	public static final double DEFAULT_SIGMA_SCALAR	= 2;
	
	public ExponentialKernel() {
		this(DEFAULT_SIGMA);
	}
	
	public ExponentialKernel(final double sigma) {
		this(sigma, DEFAULT_EXPONENTIAL);
	}
	
	/**
	 * For use with GaussianKernal
	 * @param SIGMA
	 * @param EXPONENTIAL
	 */
	protected ExponentialKernel(final double SIGMA, final double EXPONENTIAL) {
		super(SIGMA, EXPONENTIAL, DEFAULT_SIGMA_EXP, DEFAULT_SIGMA_SCALAR);
	}
	
	@Override
	public String getName() {
		return "ExponentialKernel";
	}
}
