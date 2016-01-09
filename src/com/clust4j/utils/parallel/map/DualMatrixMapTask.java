package com.clust4j.utils.parallel.map;

abstract class DualMatrixMapTask extends MatrixMapTask {
	private static final long serialVersionUID = -1965024834449661972L;
	
	final double[][] matrix_b;
	final double[][] matrix_c;

	DualMatrixMapTask(double[][] mat, double[][] mat_b, double[][] result, int lo, int hi) {
		super(mat, lo, hi);
		matrix_b = mat_b;
		matrix_c = result;
	}
}
