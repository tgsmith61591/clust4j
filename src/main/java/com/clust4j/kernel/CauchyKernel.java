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
 * The Cauchy kernel comes from the <a href="http://en.wikipedia.org/wiki/Cauchy_distribution">Cauchy distribution</a>
 * (<a href="http://figment.cse.usf.edu/~sfefilat/data/papers/WeAT4.2.pdf">Basak, 2008</a>). 
 * It is a long-tailed kernel and can be used to give long-range influence and 
 * sensitivity over the high dimension space.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class CauchyKernel extends RadialBasisKernel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7099384030117130226L;
	public CauchyKernel() {
		super();
	}
	
	public CauchyKernel(final double sigma) {
		super(sigma);
	}
	
	@Override
	public String getName() {
		return "CauchyKernel";
	}
	
	@Override
	final public double getPartialSimilarity(final double[] a, final double[] b) {
		final double lp2 = FastMath.pow(toHilbertPSpace(a, b), 2);
		return 1.0 / (1 + lp2/FastMath.pow(getSigma(), 2));
	}
	
	@Override
	final public double partialSimilarityToSimilarity(double partial) {
		return partial;
	}
	
	@Override
	final public double similarityToPartialSimilarity(double full) {
		return full;
	}
}
