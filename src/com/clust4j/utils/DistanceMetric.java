package com.clust4j.utils;

public interface DistanceMetric extends GeometricallySeparable {
	public static final int DEFAULT_P = 2;
	
	public double getP();
	
	/**
	 * If the metric allows for a faster, non-canonical computation
	 * that will maintain ordinality in distance computations,
	 * this method will compute it. Otherwise, it should return
	 * {@link #getDistance(double[], double[])}. E.g., for {@link Distance#EUCLIDEAN},
	 * the reduced distance method will not compute the sqrt as the final
	 * stage for the sake of efficiency.
	 * @param a
	 * @param b
	 * @return
	 */
	public double getReducedDistance(double[] a, double[] b);
	
	/**
	 * Convert the reduced distance to the full distance
	 * @param a
	 * @param b
	 * @return
	 */
	public double reducedDistanceToDistance(double[] a, double[] b);
}
