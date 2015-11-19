package com.clust4j.kernel;

/**
 * The Inverse {@link MultiQuadricKernel}. As with the {@link GaussianKernel}, 
 * it results in a kernel matrix with full rank 
 * <a href="http://www.springerlink.com/content/w62233k766460945/">(Micchelli, 1986)</a> and thus 
 * forms an infinite dimension feature space.
 * 
 * <p>If <tt>||<i>x</i> - <i>y</i>|| < SIGMA</tt>, the similarity is 0.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class InverseMultiquadricKernel extends MultiquadricKernel {
	
	public InverseMultiquadricKernel() {
		super();
	}
	
	public InverseMultiquadricKernel(final double constant) {
		super(constant);
	}
	
	@Override
	public String getName() {
		return "InverseMultiquadricKernel";
	}
	
	@Override
	public double getSeparability(final double[] a, final double[] b) {
		return 1 / super.getSeparability(a, b);
	}
}
