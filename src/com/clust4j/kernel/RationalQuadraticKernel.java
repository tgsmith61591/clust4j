package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Rational Quadratic kernel is less computationally 
 * intensive than the {@link GaussianKernel} and can be used as an 
 * alternative when using the Gaussian becomes too expensive.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class RationalQuadraticKernel extends ConstantKernel {
	private final double constant;
	
	public RationalQuadraticKernel() { this(DEFAULT_CONSTANT); }
	public RationalQuadraticKernel(final double constant) {
		super();
		this.constant = constant;
	}
	
	
	@Override
	public double getSimilarity(double[] a, double[] b) {
		final double lp = toHilbertPSpace(a, b);
		final double sqnm = FastMath.pow(lp, 2);
		return 1 - (sqnm / (sqnm + getConstant()));
	}

	@Override
	public double getConstant() {
		return constant;
	}

	@Override
	public String getName() {
		return "RationalQuadraticKernel";
	}
}
