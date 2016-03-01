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
}
