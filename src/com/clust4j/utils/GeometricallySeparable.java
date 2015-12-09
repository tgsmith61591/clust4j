package com.clust4j.utils;

public interface GeometricallySeparable extends java.io.Serializable {
	/**
	 * Generally equal to negative {@link #getSimilarity(double[], double[])}
	 * @param a
	 * @param b
	 * @return
	 */
	public double getDistance(final double[] a, final double[] b);
	public String getName();
}
