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
	protected final double constant;
	protected final double degree;
	
	public PolynomialKernel() {
		this(DEFAULT_DEGREE, DEFAULT_ALPHA, DEFAULT_CONSTANT);
	}
	
	public PolynomialKernel(final double degree, final double alpha) {
		this(degree, alpha, DEFAULT_CONSTANT);
	}
	
	public PolynomialKernel(final double degree, final double alpha, final double constant) {
		this.degree = degree;
		this.alpha = alpha;
		this.constant = constant;
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
	public double getConstant() {
		return constant;
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		return FastMath.pow(alpha * VecUtils.innerProductForceSerial(a, b) + getConstant(), degree);
	}
}
