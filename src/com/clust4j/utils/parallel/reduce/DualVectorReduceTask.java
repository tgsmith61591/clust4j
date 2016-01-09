package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

abstract class DualVectorReduceTask<T> extends VectorReduceTask<T> {
	private static final long serialVersionUID = -4647929958194428774L;
	
	final double[] array_b;
	
	DualVectorReduceTask(double[] arr, double[] arr_b, int lo, int hi) {
		super(arr, lo, hi);
		
		VecUtils.checkDims(arr, arr_b); // If equal, result is ok length too
		array_b = arr_b;
	}

}
