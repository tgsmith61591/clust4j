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
 * The Rational Quadratic kernel is less computationally 
 * intensive than the {@link GaussianKernel} and can be used as an 
 * alternative when using the Gaussian becomes too expensive.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class RationalQuadraticKernel extends ConstantKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7063644380491570720L;
	
	public RationalQuadraticKernel() { this(DEFAULT_CONSTANT); }
	public RationalQuadraticKernel(final double constant) {
		super(constant);
	}
	
	
	@Override
	public double getSimilarity(double[] a, double[] b) {
		final double lp = toHilbertPSpace(a, b);
		final double sqnm = FastMath.pow(lp, 2);
		return 1 - (sqnm / (sqnm + getConstant()));
	}

	@Override
	public String getName() {
		return "RationalQuadraticKernel";
	}
}
