package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The circular kernel is used in geostatic applications. 
 * It is an example of an isotropic stationary kernel 
 * and is positive definite in R<sup>2</sup>.
 * 
 * <p>If <tt>||<i>x</i> - <i>y</i>|| < SIGMA</tt>, zero otherwise.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class CircularKernel extends RadialBasisKernel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2111174336601201084L;

	public CircularKernel() { super(); }
	public CircularKernel(final double sigma) {
		super(sigma);
	}
	
	@Override
	public double getSimilarity(double[] a, double[] b) {
		final double lp = toHilbertPSpace(a, b);
		
		// Per corner case condition
		if(lp >= getSigma())
			return 0d;
		
		final double twoOverPi = (2d/FastMath.PI);
		final double lpOverSig = lp/getSigma();
		
		/* Front segment */
		final double front = twoOverPi * FastMath.acos(-lpOverSig);
		
		/* Back segment */
		final double first = twoOverPi * lpOverSig;
		final double second = FastMath.sqrt(1 - FastMath.pow(lpOverSig, 2));
		final double back = first * second;
		
		return front - back;
	}

	@Override
	public String getName() {
		return "CircularKernel";
	}
}
