package com.clust4j.metrics;

import org.apache.commons.math3.exception.DimensionMismatchException;

class ConfMatUtils {
	private static void checkDims(int[] a, int[] b) {
		if(a.length != b.length) // Allow empty; so we don't use VecUtils
			throw new DimensionMismatchException(a.length, b.length);
	}
	
	public static int numEqual(int[] a, int[] b) {
		checkDims(a, b);
		int sum = 0;
		for(int i = 0; i < a.length; i++)
			if(a[i] == b[i])
				sum++;
		return sum;
	}
	
	// TODO tp/fp/tn/fn for multiclass...
}
