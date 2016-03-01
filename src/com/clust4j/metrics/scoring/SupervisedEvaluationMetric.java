package com.clust4j.metrics.scoring;

public interface SupervisedEvaluationMetric extends java.io.Serializable {
	public double evaluate(int[] actual, int[] predicted);
}
