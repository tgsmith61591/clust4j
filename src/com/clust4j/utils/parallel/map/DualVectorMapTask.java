package com.clust4j.utils.parallel.map;

abstract class DualVectorMapTask extends VectorMapTask {
	private static final long serialVersionUID = -4647929958194428774L;
	
	final double[] array_b;
	final double[] array_c;
	
	DualVectorMapTask(double[] arr, double[] arr_b, double[] result, int lo, int hi) {
		super(arr, lo, hi);
		
		dimCheck(arr, arr_b); // If equal, result is ok length too
		array_b = arr_b;
		array_c = result;
	}

	/**
	 * Different tasks may allow empty arrays, others may not.
	 * @param a
	 * @param b
	 */
	abstract void dimCheck(double[] a, double[] b);
}
