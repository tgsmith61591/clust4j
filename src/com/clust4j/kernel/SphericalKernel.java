package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The spherical kernel is similar to the {@link CircularKernel}, 
 * but is positive definite in R<sup>3</sup>.
 * 
 * <p>If <tt>||<i>x</i> - <i>y</i>|| < SIGMA</tt>, zero otherwise.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class SphericalKernel extends CircularKernel {

	public SphericalKernel() {
		super();
	}
	
	public SphericalKernel(final double sigma) {
		super(sigma);
	}
	
	@Override
	public String getName() {
		return "SphericalKernel";
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		final double lp = toHilbertPSpace(a, b);
		if(lp >= getSigma())
			return 0d;
		
		final double lpOverSig = lp / getSigma();
		final double front = 1 - 1.5 * lpOverSig;
		final double back = 0.5 * FastMath.pow(lpOverSig, 3);
		
		return front + back;
	}
}
