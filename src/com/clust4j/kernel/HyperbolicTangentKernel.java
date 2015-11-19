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
	public static final double DEFAULT_ALPHA = 1;
	
	private final double constant;
	private final double alpha;
	
	
	public HyperbolicTangentKernel() { this(DEFAULT_CONSTANT, DEFAULT_ALPHA); }
	public HyperbolicTangentKernel(final double constant, final double alpha) {
		super();
		this.constant = constant;
		this.alpha = alpha;
	}
	
	
	
	@Override
	public double getSimilarity(double[] a, double[] b) {
		return FastMath.tanh(alpha * VecUtils.innerProduct(a, b) + getConstant());
	}

	@Override
	public String getName() {
		return "HyperbolicTangent (Sigmoid) Kernel";
	}
	
	public double getAlpha() {
		return alpha;
	}

	@Override
	public double getConstant() {
		return constant;
	}

}
