package com.clust4j.metrics.scoring;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;

public interface UnsupervisedEvaluationMetric {
	public double evaluate(AbstractClusterer model, int[] labels);
	public double evaluate(AbstractRealMatrix mat, int[] labels);
}
