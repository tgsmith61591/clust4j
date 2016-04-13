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
 * The Multiquadric kernel can be used in the same situations as the {@link RationalQuadraticKernel}. 
 * As is the case with the Sigmoid kernel ({@link HyperbolicTangentKernel}), it is also an example of an non-positive definite kernel.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class MultiquadricKernel extends ConstantKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3023302397706144064L;
	
	public MultiquadricKernel() { this(DEFAULT_CONSTANT); }
	public MultiquadricKernel(final double constant) {
		super(constant);
	}

	@Override
	public String getName() {
		return "MultiquadricKernel";
	}

	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		double lp = toHilbertPSpace(a, b);
		double sqnm = FastMath.pow(lp, 2);
		return FastMath.sqrt(sqnm + FastMath.pow(getConstant(), 2));
	}
}
