package com.clust4j.kernel;

import com.clust4j.utils.VecUtils;

/**
 * The Histogram Intersection Kernel is also known as the 
 * Min Kernel and has been proven useful in image classification.
 *
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class MinKernel extends Kernel {
	
	public MinKernel() {
		super();
	}

	@Override
	public double getSimilarity(double[] a, double[] b) {
		return VecUtils.sum(VecUtils.pmin(a, b));
	}

	@Override
	public String getName() {
		return "Min (Histogram Intersection) Kernel";
	}

}
