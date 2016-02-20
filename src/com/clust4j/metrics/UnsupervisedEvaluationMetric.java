package com.clust4j.metrics;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.utils.GeometricallySeparable;

public interface UnsupervisedEvaluationMetric {
	public double evaluate(AbstractClusterer model, GeometricallySeparable metric, int[] labels);
	public double evaluate(AbstractRealMatrix mat, GeometricallySeparable metric, int[] labels);
}
