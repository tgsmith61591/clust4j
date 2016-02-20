package com.clust4j.metrics;

public interface SupervisedEvaluationMetric extends java.io.Serializable {
	public double evaluate(int[] actual, int[] predicted);
}
