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
 * The Hyperbolic Tangent Kernel, also known as the 
 * Sigmoid Kernel and as the Multilayer Perceptron (MLP) 
 * kernel, comes from the Neural Networks field, where 
 * the bipolar sigmoid function is often used as an 
 * activation function for artificial neurons.
 * 
 * <p>It is interesting to note that a SVM model using a 
 * sigmoid kernel function is equivalent to a two-layer, 
 * perceptron neural network. This kernel was quite popular 
 * for support vector machines due to its origin from neural 
 * network theory. Also, despite being only conditionally 
 * positive definite, it has been found to perform well 
 * in practice.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class HyperbolicTangentKernel extends ConstantKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2362070006438269124L;

	public static final double DEFAULT_ALPHA = 1.0;
	private final double alpha;
	
	
	public HyperbolicTangentKernel() { this(DEFAULT_CONSTANT, DEFAULT_ALPHA); }
	public HyperbolicTangentKernel(final double constant, final double alpha) {
		super(constant);
		this.alpha = alpha;
	}
	
	
	// We can't compute a partial similarity for tanh because it will lose ordinality
	@Override
	public double getSimilarity(double[] a, double[] b) {
		return FastMath.tanh(getAlpha() * VecUtils.innerProduct(a, b) + getConstant());
	}

	@Override
	public String getName() {
		return "Sigmoid (tanh) Kernel";
	}
	
	public double getAlpha() {
		return alpha;
	}
}
