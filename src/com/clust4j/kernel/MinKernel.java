package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

/**
 * The Histogram Intersection Kernel is also known as the 
 * Min Kernel and has been proven useful in image classification.
 *
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class MinKernel extends AbstractKernel {
	
	public MinKernel() {
		super();
	}

	@Override
	public double distance(double[] a, double[] b) {
		VecUtils.checkDims(a, b); // Make sure same len...
		
		double sum = 0;
		for(int i = 0; i < a.length; i++)
			sum += FastMath.min(a[i], b[i]);
		
		return sum;
	}

	@Override
	public String getName() {
		return "Min (Histogram Intersection) Kernel";
	}

}
