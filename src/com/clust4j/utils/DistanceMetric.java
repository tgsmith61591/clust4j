package com.clust4j.utils;

public interface DistanceMetric extends GeometricallySeparable {
	public static final double DEFAULT_P = 2.0;
	
	public double getP();
	
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
	 * @return
	 */
	public double getPartialDistance(double[] a, double[] b);
	
	/**
	 * Convert the partial distance to the full distance
	 * @param a
	 * @param b
	 * @return
	 */
	public double partialDistanceToDistance(double d);
	
	/**
	 * Convert the full distance to the partial distance
	 * @param a
	 * @param b
	 * @return
	 */
	public double distanceToPartialDistance(double d);
}
