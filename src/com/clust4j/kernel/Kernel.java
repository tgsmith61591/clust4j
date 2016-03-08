package com.clust4j.kernel;

import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.VecUtils;

/**
 * Highest level of kernel abstraction. For kernels whose similarity
 * may return {@link Double#NaN}, return {@link Double#NEGATIVE_INFINITY},
 * as kernels are a similarity metric and should minimize similarity in these
 * instances.
 * 
 * @author Taylor G Smith
 */
public abstract class Kernel implements SimilarityMetric {
	private static final long serialVersionUID = -630865804908845073L;


	public Kernel() {}
	
	
	
	@Override
	public double getDistance(final double[] a, final double[] b) {
		return -getSimilarity(a, b);
	}
	
	@Override
	public double getPartialDistance(final double[] a, final double[] b) {
		return -getPartialSimilarity(a, b);
	}
	
	@Override
	public double getPartialSimilarity(final double[] a, final double[] b) {
		return getSimilarity(a, b);
	}
	
	@Override
	public double distanceToPartialDistance(double d) {
		return -similarityToPartialSimilarity(-d);
	}
	
	@Override
	public double similarityToPartialSimilarity(double d) {
		return d;
	}
	
	@Override
	public double partialDistanceToDistance(double d) {
		return -partialSimilarityToSimilarity(-d);
	}
	
	@Override
	public double partialSimilarityToSimilarity(double d) {
		return d;
	}
	
	
	
	protected static double toHilbertPSpace(final double[] a, final double[] b) {
		// Originally: 2*VecUtils.innerProductForceSerial(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
		// This costs 3N!!
		VecUtils.checkDims(a,b);
		double ipab = 0, ipaa = 0, ipbb = 0;
		int n = a.length;
		
		// This only costs 1N but is uglier...
		for(int i = 0; i < n; i++) {
			ipab += a[i] * b[i];
			ipaa += a[i] * a[i];
			ipbb += b[i] * b[i];
		}
		
		//return 2*VecUtils.innerProductForceSerial(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
		return 2*ipab - ipaa - ipbb;
	}
	
	/**
	 * Returns the name of the kernel
	 */
	@Override
	public String toString() {
		return getName();
	}
}
