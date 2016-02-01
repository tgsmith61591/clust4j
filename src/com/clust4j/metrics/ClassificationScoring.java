package com.clust4j.metrics;

public enum ClassificationScoring implements EvaluationMetric {
	ACCURACY {
		@Override
		public double evaluate(final int[] actual, final int[] predicted) {
			return ConfMatUtils.numEqual(actual, predicted) / (double)actual.length;
		}
	}
}
