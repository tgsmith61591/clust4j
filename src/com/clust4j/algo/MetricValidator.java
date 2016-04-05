package com.clust4j.algo;

import com.clust4j.metrics.pairwise.GeometricallySeparable;

public interface MetricValidator {
	public boolean isValidMetric(GeometricallySeparable geo);
}
