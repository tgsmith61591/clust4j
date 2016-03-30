package com.clust4j.utils.parallel.reduce;

import org.apache.commons.math3.exception.DimensionMismatchException;

@Deprecated
abstract class DualVectorReduceTask<T> extends VectorReduceTask<T> {
	private static final long serialVersionUID = -4647929958194428774L;
	
	final double[] array_b;
	
	DualVectorReduceTask(double[] arr, double[] arr_b, int lo, int hi) {
		super(arr, lo, hi);
		checkDims(arr_b); // super class handles arr
		
		if(arr.length != arr_b.length)
			throw new DimensionMismatchException(arr.length, arr_b.length);
		array_b = arr_b;
	}

}
