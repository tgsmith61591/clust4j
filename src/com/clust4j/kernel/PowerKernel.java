package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Power kernel is also known as the (unrectified) triangular kernel. 
 * It is an example of scale-invariant kernel <a href="http://hal.archives-ouvertes.fr/docs/00/07/19/84/PDF/RR-4601.pdf">
 * (Sahbi and Fleuret, 2004)</a> and is also only conditionally positive definite.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class PowerKernel extends Kernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -861680950436032350L;
	public static final double DEFAULT_DEGREE = 1;
	private final double degree;
	
	public PowerKernel() {
		this(DEFAULT_DEGREE);
	}
	
	public PowerKernel(final double degree) {
		this.degree = degree;
	}

	@Override
	public double getSimilarity(double[] a, double[] b) {
		return -(FastMath.pow(toHilbertPSpace(a, b), degree));
	}

	@Override
	public String getName() {
		return "PowerKernel";
	}

	public double getDegree() {
		return degree;
	}
}
