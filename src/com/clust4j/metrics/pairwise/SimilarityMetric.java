package com.clust4j.metrics.pairwise;

public interface SimilarityMetric extends GeometricallySeparable, java.io.Serializable { 
	/**
	 * Generally equal to negative {@link #getDistance(double[], double[])}
	 * @param a
	 * @param b
	 * @return
	 */
	public double getSimilarity(final double[] a, final double[] b);
}
