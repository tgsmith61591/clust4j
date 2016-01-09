package com.clust4j.utils.parallel.map;

import com.clust4j.utils.parallel.MatrixMRTask;

abstract class MatrixMapTask extends MatrixMRTask<double[][]> {
	private static final long serialVersionUID = -8682414038605706202L;

	MatrixMapTask(double[][] mat, int lo, int hi) {
		super(mat, lo, hi);
	}
}
