package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Multiquadric kernel can be used in the same situations as the {@link RationalQuadraticKernel}. 
 * As is the case with the Sigmoid kernel ({@link HyperbolicTangentKernel}), it is also an example of an non-positive definite kernel.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class MultiquadricKernel extends AbstractConstantKernel {
	private final double constant;
	
	public MultiquadricKernel() { this(DEFAULT_CONSTANT); }
	public MultiquadricKernel(final double constant) {
		this.constant = constant;
	}

	@Override
	public String getName() {
		return "MultiquadricKernel";
	}

	@Override
	public double getConstant() {
		return constant;
	}

	@Override
	public double distance(final double[] a, final double[] b) {
		double lp = getLpNorm(a, b, 2);
		double sqnm = FastMath.pow(lp, 2);
		return FastMath.sqrt(sqnm + FastMath.pow(getConstant(), 2));
	}
}
