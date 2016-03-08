package com.clust4j.metrics.pairwise;

import com.clust4j.utils.Named;

public interface GeometricallySeparable extends java.io.Serializable, Named {
	
	/**
	 * Generally equal to negative {@link #getSimilarity(double[], double[])}
	 * @param a
	 * @param b
	 * @return
	 */
	public double getDistance(final double[] a, final double[] b);
	
	/**
	 * If the metric allows for a faster, non-canonical computation
	 * that will maintain ordinality in distance computations,
	 * this method will compute it. Otherwise, it should return
	 * {@link #getDistance(double[], double[])}. 
	 * 
	 * E.g., for {@link Distance#EUCLIDEAN},
	 * the partial distance method will not compute the sqrt as the final
	 * stage for the sake of efficiency.
	 * @param a
	 * @param b
	 * @return the partial distance
	 */
	public double getPartialDistance(final double[] a, final double[] b);
	
	/**
	 * Convert the partial distance to the full distance
	 * @param a
	 * @param b
	 * @return the full distance
	 */
	public double partialDistanceToDistance(double d);
	
	/**
	 * Convert the full distance to the partial distance
	 * @param a
	 * @param b
	 * @return the partial distance
	 */
	public double distanceToPartialDistance(double d);
}
