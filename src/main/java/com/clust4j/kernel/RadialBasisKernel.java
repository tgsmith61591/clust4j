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

/**
 * Implementation of the radial basis kernel function. 
 * The adjustable parameter sigma plays a major role in the performance 
 * of the kernel, and should be carefully tuned to the problem at hand. 
 * If overestimated, the exponential will behave almost linearly and 
 * the higher-dimensional projection will start to lose its non-linear 
 * power. In the other hand, if underestimated, the function will lack 
 * regularization and the decision boundary will be highly sensitive to 
 * noise in training data.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class RadialBasisKernel extends Kernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3281494130468137896L;
	public final static double DEFAULT_SIGMA = 1;
	private final double sigma;
	
	public RadialBasisKernel() { this(DEFAULT_SIGMA); }
	public RadialBasisKernel(final double sigma) {
		this.sigma = sigma;
	}
	
	@Override
	public String getName() {
		return "RadialKernel";
	}
	
	public double getSigma() {
		return sigma;
	}
	
	@Override
	final public double getSimilarity(double[] a, double[] b) {
		return partialSimilarityToSimilarity(getPartialSimilarity(a, b));
	}
	
	@Override
	public double getPartialSimilarity(final double[] a, final double[] b) {
		return sigma * toHilbertPSpace(a,b);
	}
	
	@Override
	public double partialSimilarityToSimilarity(double partial) {
		return FastMath.exp(partial);
	}
	
	@Override
	public double similarityToPartialSimilarity(double full) {
		return FastMath.log(full);
	}
}
