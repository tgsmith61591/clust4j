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
	public double getPartialSimilarity(double[] a, double[] b) {
		final double lp = toHilbertPSpace(a, b);
		
		// Per corner case condition
		if(lp >= getSigma())
			return 0.0;
		
		final double twoOverPi = (2d/FastMath.PI);
		final double lpOverSig = lp/getSigma();
		
		/* Front segment */
		final double front = twoOverPi * FastMath.acos(-lpOverSig);
		
		/* Back segment */
		final double first = twoOverPi * lpOverSig;
		final double second = FastMath.sqrt(1.0 - FastMath.pow(lpOverSig, 2));
		final double back = first * second;
		final double answer = front - back;
		
		return Double.isNaN(answer) ? Double.NEGATIVE_INFINITY : answer;
	}
	
	@Override
	final public double partialSimilarityToSimilarity(double partial) {
		return partial;
	}
	
	@Override
	final public double similarityToPartialSimilarity(double full) {
		return full;
	}

	@Override
	public String getName() {
		return "CircularKernel";
	}
}
