package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

/**
 * The ANOVA kernel is also a {@link RadialBasisKernel}, just as the {@link GaussianKernel} 
 * and {@link LaplacianKernel}. It is said to perform well in multidimensional 
 * regression problems <a href="http://www.nicta.com.au/research/research_publications?sq_content_src=%2BdXJsPWh0dHBzJTNBJTJGJTJGcHVibGljYXRpb25zLmluc2lkZS5uaWN0YS5jb20uYXUlMkZzZWFyY2glMkZmdWxsdGV4dCUzRmlkJTNEMjYxJmFsbD0x">(Hofmann, 2008)</a>.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class ANOVAKernel extends RadialBasisKernel {
	final public static double DEFAULT_DEGREE = 1;
	private final double degree;

	public ANOVAKernel() {
		this(DEFAULT_DEGREE);
	}
	
	public ANOVAKernel(final double degree) {
		this(DEFAULT_SIGMA, degree);
	}
	
	public ANOVAKernel(final double sigma, final double degree) {
		super(sigma);
		
		this.degree = degree;
	}
	
	public double getDegree() {
		return degree;
	}
	
	@Override
	public String getName() {
		return "ANOVAKernel";
	}
	
	@Override
	public double getSeparability(final double[] a, final double[] b) {
		final double[] xMinY2 = VecUtils.pow( VecUtils.subtract(a, b), 2 );
		final double[] sigmaXY2 = VecUtils.scalarMultiply(xMinY2, -getSigma());
		
		double sum = 0;
		for(double d: sigmaXY2)
			sum += FastMath.exp(d);
		
		return sum;
	}
}
