package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * Implementation of the radial basis kernel function. 
 * The adjustable parameter sigma plays a major role in the performance 
 * of the kernel, and should be carefully tuned to the problem at hand. 
 * If overestimated, the exponential will behave almost linearly and 
 * the higher-dimensional projection will start to lose its non-linear 
 * power. In the other hand, if underestimated, the function will lack 
 * regularization and the decision boundary will be highly sensitive to 
 * noise in training data.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class RadialBasisKernel extends Kernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3281494130468137896L;
	public final static double DEFAULT_SIGMA = 1;
	private double sigma;
	
	public RadialBasisKernel() { this(DEFAULT_SIGMA); }
	public RadialBasisKernel(final double sigma) {
		this.sigma = sigma;
	}
	
	@Override
	public String getName() {
		return "RadialKernel";
	}
	
	public double getSigma() {
		return sigma;
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		return FastMath.exp( sigma * toHilbertPSpace(a,b) );
	}
	
	public void setSigma(final double sigma) {
		this.sigma = sigma;
	}
}
