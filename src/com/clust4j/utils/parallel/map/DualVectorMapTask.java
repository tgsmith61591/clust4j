package com.clust4j.utils.parallel.map;

import com.clust4j.utils.VecUtils;

abstract class DualVectorMapTask extends VectorMapTask {
	private static final long serialVersionUID = -4647929958194428774L;
	
	final double[] array_b;
	final double[] array_c;
	
	DualVectorMapTask(double[] arr, double[] arr_b, double[] result, int lo, int hi) {
		super(arr, lo, hi);
		
		VecUtils.checkDims(arr, arr_b); // If equal, result is ok length too
		array_b = arr_b;
		array_c = result;
	}

}
