package com.clust4j.utils.parallel.map;
import com.clust4j.utils.parallel.VectorMRTask;

abstract class VectorMapTask extends VectorMRTask<double[]> {
	private static final long serialVersionUID = -7986981765361158408L;
	
	VectorMapTask(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}
}
