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
 * The ANOVA kernel is also a {@link RadialBasisKernel}, just as the {@link GaussianKernel} 
 * and {@link LaplacianKernel}. It is said to perform well in multidimensional 
 * regression problems <a href="http://www.nicta.com.au/research/research_publications?sq_content_src=%2BdXJsPWh0dHBzJTNBJTJGJTJGcHVibGljYXRpb25zLmluc2lkZS5uaWN0YS5jb20uYXUlMkZzZWFyY2glMkZmdWxsdGV4dCUzRmlkJTNEMjYxJmFsbD0x">(Hofmann, 2008)</a>.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class ANOVAKernel extends RadialBasisKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2352083487381024145L;
	final public static double DEFAULT_DEGREE = 1;
	private final double degree;

	public ANOVAKernel() {
		this(DEFAULT_DEGREE);
	}
	
	public ANOVAKernel(final double degree) {
		this(DEFAULT_SIGMA, degree);
	}
	
	public ANOVAKernel(final double sigma, final double degree) {
		super(sigma);
		
		this.degree = degree;
	}
	
	public double getDegree() {
		return degree;
	}
	
	@Override
	public String getName() {
		return "ANOVAKernel";
	}
	
	@Override
	public double getPartialSimilarity(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		
		double s = 0, diff;
		for(int i = 0; i < a.length; i++) {
			diff = a[i] - b[i];
			s += FastMath.pow(FastMath.exp((diff * diff) * -getSigma()), getDegree());
		}
		
		return s;
	}
	
	@Override
	public double partialSimilarityToSimilarity(double partial) {
		return partial;
	}
	
	@Override
	public double similarityToPartialSimilarity(double full) {
		return full;
	}
}
