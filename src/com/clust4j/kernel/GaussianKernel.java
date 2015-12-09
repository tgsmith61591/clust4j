package com.clust4j.kernel;

/**
 * The Gaussian kernel is an example of radial basis function kernel:
 * 
 * <p><code>k(x,y) = exp(-||x-y||<sup>2</sup>/2*sigma<sup>2</sup>)</code></p>
 * 
 * The adjustable parameter sigma plays a major role in the performance of the kernel, 
 * and should be carefully tuned to the problem at hand. If overestimated, the exponential 
 * will behave almost linearly and the higher-dimensional projection will start to lose 
 * its non-linear power. In the other hand, if underestimated, the function will lack 
 * regularization and the decision boundary will be highly sensitive to noise in training data.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class GaussianKernel extends ExponentialKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3764791479335863828L;
	public static final double DEF_EXP = 2;
	
	public GaussianKernel() {
		this(DEFAULT_SIGMA);
	}
	
	public GaussianKernel(final double sigma) {
		super(sigma, DEF_EXP);
	}
	
	@Override
	public String getName() {
		return "GaussianKernel";
	}
}
