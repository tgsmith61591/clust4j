package com.clust4j.metrics.scoring;

import org.apache.commons.math3.exception.DimensionMismatchException;

public enum BinomialClassificationScoring implements SupervisedEvaluationMetric {
	ACCURACY {
		@Override
		public double evaluate(final int[] actual, final int[] predicted) {
			return numEqual(actual, predicted) / (double)actual.length;
		}
	},

	// TODO: more...
	;
	
	private static void checkDims(int[] a, int[] b) {
		if(a.length != b.length) // Allow empty; so we don't use VecUtils
			throw new DimensionMismatchException(a.length, b.length);
	}
	
	private static int numEqual(int[] a, int[] b) {
		checkDims(a, b);
		int sum = 0;
		for(int i = 0; i < a.length; i++)
			if(a[i] == b[i])
				sum++;
		return sum;
	}
	
	// TODO: tp/fp/tn/fn for multiclass...
}
