package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Laplace Kernel is completely equivalent to the exponential kernel, 
 * except for being less sensitive for changes in the sigma parameter. 
 * Being equivalent, it is also a radial basis function kernel.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class LaplacianKernel extends RadialBasisKernel {
	public static final double DEFAULT_EXPONENTIAL	= 1;
	public static final double DEFAULT_SIGMA_EXP	= 1;
	public static final double DEFAULT_SIGMA_SCALAR	= 1;

	protected final double exponential;
	protected final double sigma_exp;
	protected final double sigma_scalar;
	
	
	public LaplacianKernel() {
		this(DEFAULT_SIGMA, DEFAULT_EXPONENTIAL, DEFAULT_SIGMA_EXP, DEFAULT_SIGMA_SCALAR);
	}
	
	public LaplacianKernel(final double sigma) {
		this(sigma, DEFAULT_EXPONENTIAL, DEFAULT_SIGMA_EXP, DEFAULT_SIGMA_SCALAR);
	}
	
	public LaplacianKernel(final double sigma, final double exponential) {
		this(sigma, exponential, DEFAULT_SIGMA_EXP, DEFAULT_SIGMA_SCALAR);
	}
	
	public LaplacianKernel(final double sigma, final double exponential, final double sigma_exp) {
		this(sigma, exponential, sigma_exp, DEFAULT_SIGMA_SCALAR);
	}
	
	public LaplacianKernel(final double sigma, final double exponential, 
			final double sigma_exp, final double sigma_scalar) {
		super(sigma);
		this.exponential = exponential;
		this.sigma_exp = sigma_exp;
		this.sigma_scalar = sigma_scalar;
	}
	

	@Override
	public double getSeparability(double[] a, double[] b) {
		// Kernlab's laplacedot returns:
		// return(exp(-sigma*sqrt(-(round(2*crossprod(x,y) - crossprod(x) - crossprod(y),9)))))
		//
		// which simplifies to:
		// return(exp(-sigma*sqrt(-hilbert)))
		
		
		double hilbert = toHilbertPSpace(a, b);
		hilbert = exponential > 1 ? FastMath.pow(hilbert, exponential) : -hilbert;
		final double sigma_val = sigma_scalar * FastMath.pow(getSigma(), sigma_exp);
		
		return FastMath.exp(-sigma_val * FastMath.sqrt(hilbert));
	}
	
	public double getPower() {
		return exponential;
	}
	
	public double getSigmaPower() {
		return sigma_exp;
	}

	public double getSigmaScalar() {
		return sigma_scalar;
	}
	
	
	@Override
	public String getName() {
		return "LaplacianKernel";
	}
	
}
