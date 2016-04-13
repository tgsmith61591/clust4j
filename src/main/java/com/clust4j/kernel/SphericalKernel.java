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
 * The spherical kernel is similar to the {@link CircularKernel}, 
 * but is positive definite in R<sup>3</sup>.
 * 
 * <p>If <tt>||<i>x</i> - <i>y</i>|| < SIGMA</tt>, zero otherwise.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class SphericalKernel extends CircularKernel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4173771493103734665L;

	public SphericalKernel() {
		super();
	}
	
	public SphericalKernel(final double sigma) {
		super(sigma);
	}
	
	@Override
	public String getName() {
		return "SphericalKernel";
	}
	
	@Override
	public double getPartialSimilarity(final double[] a, final double[] b) {
		final double lp = toHilbertPSpace(a, b);
		if(lp >= getSigma())
			return 0.0;
		
		final double lpOverSig = lp / getSigma();
		final double front = 1 - 1.5 * lpOverSig;
		final double back = 0.5 * FastMath.pow(lpOverSig, 3);
		
		return front + back;
	}
}
