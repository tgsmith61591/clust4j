package com.clust4j.kernel;

import com.clust4j.utils.VecUtils;

/**
 * The Linear kernel is the simplest kernel function. 
 * It is given by the inner product <x,y> plus an optional constant c. 
 * Kernel algorithms using a linear kernel are often equivalent to 
 * their non-kernel counterparts, i.e. 
 * 
 * <a href="http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/2010/01/kernel-principal-component-analysis-in-c/">kernel principal component analysis</a> 
 * with linear kernel is the same as standard PCA.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class LinearKernel extends AbstractConstantKernel {
	private double constant;

	public LinearKernel() { this(DEFAULT_CONSTANT); }
	public LinearKernel(final double constant) {
		super();
		this.constant = constant;
	}
	
	@Override
	public double distance(final double[] a, final double[] b) {
		return VecUtils.innerProduct(a, b) + getConstant();
	}
	
	@Override
	public String getName() {
		return "LinearKernel";
	}

	@Override
	public double getConstant() {
		return constant;
	}
}
