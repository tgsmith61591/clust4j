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
 * The Polynomial kernel is a non-stationary kernel. 
 * Polynomial kernels are well suited for problems 
 * where all the training data is normalized.
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class PolynomialKernel extends ConstantKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7356583309481333635L;
	public final static double DEFAULT_ALPHA = 1;
	public final static double DEFAULT_DEGREE= 1;
	
	protected final double alpha;
	protected final double degree;
	
	public PolynomialKernel() {
		this(DEFAULT_DEGREE, DEFAULT_ALPHA, DEFAULT_CONSTANT);
	}
	
	public PolynomialKernel(final double degree, final double alpha) {
		this(degree, alpha, DEFAULT_CONSTANT);
	}
	
	public PolynomialKernel(final double degree, final double alpha, final double constant) {
		super(constant);
		this.degree = degree;
		this.alpha = alpha;
	}
	
	public double getAlpha() {
		return alpha;
	}
	
	public double getDegree() {
		return degree;
	}

	@Override
	public String getName() {
		return "PolynomialKernel";
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		return FastMath.pow(getAlpha() * VecUtils.innerProduct(a, b) + getConstant(), getDegree());
	}
}
