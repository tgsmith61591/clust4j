package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

/**
 * The Generalized Histogram Intersection kernel 
 * (GeneralizedMinKernel) is built based on the {@link MinKernel} for image 
 * classification but applies in a much larger variety of 
 * contexts <a href="http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf">(Boughorbel, 2005)</a>. It is given by:
 *
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class GeneralizedMinKernel extends MinKernel {
	private double alpha;
	private double beta;
	
	public GeneralizedMinKernel(final double alpha, final double beta) {
		super();
	}
	
	public double getAlpha() {
		return alpha;
	}
	
	public double getBeta() {
		return beta;
	}
	
	@Override
	public String getName() {
		return "GeneralizedMin (Generalized Histogram Intersection) Kernel";
	}
	
	@Override
	public double getSeparability(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		
		double sum = 0;
		for(int i = 0; i < a.length; i++)
			sum += FastMath.min( FastMath.pow(FastMath.abs(a[i]), alpha), 
								 FastMath.pow(FastMath.abs(b[i]), beta) );
		
		return sum;
	}
}
