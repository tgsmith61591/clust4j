package com.clust4j.metrics.pairwise;

import com.clust4j.metrics.pairwise.MinkowskiDistance;

public interface DistanceMetric extends GeometricallySeparable {
	public static final double DEFAULT_P = 2.0;
	
	/**
	 * Get the p parameter for the distance metric
	 * @see {@link MinkowskiDistance}
	 * @return the p parameter
	 */
	public double getP();
}
