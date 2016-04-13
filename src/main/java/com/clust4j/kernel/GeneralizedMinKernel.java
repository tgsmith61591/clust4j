/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
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
	/**
	 * 
	 */
	private static final long serialVersionUID = -3798280254415501176L;
	public static final double DEF_ALPHA = 1.0;
	public static final double DEF_BETA = 1.0;
	final private double alpha;
	final private double beta;
	
	public GeneralizedMinKernel() {
		this(DEF_ALPHA, DEF_BETA);
	}
	
	public GeneralizedMinKernel(final double alpha, final double beta) {
		super();
		this.alpha = alpha;
		this.beta = beta;
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
	public double getSimilarity(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		
		double sum = 0;
		for(int i = 0; i < a.length; i++)
			sum += FastMath.min( FastMath.pow(FastMath.abs(a[i]), getAlpha()), 
								 FastMath.pow(FastMath.abs(b[i]), getBeta()));
		
		return sum;
	}
}
