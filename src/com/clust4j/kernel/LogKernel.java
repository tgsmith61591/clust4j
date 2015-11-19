package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Log kernel seems to be particularly interesting for 
 * images, but is only conditionally positive definite
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class LogKernel extends PowerKernel {

	public LogKernel() {
		super();
	}
	
	public LogKernel(final double degree) {
		super(degree);
	}
	
	@Override
	public String getName() {
		return "LogKernel";
	}
	
	@Override
	public double getSeparability(final double[] a, final double[] b) {
		final double sup = super.getSeparability(a, b);
		return -FastMath.log(-sup + 1);
	}
}
