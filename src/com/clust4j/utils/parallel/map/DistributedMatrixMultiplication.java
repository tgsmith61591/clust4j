package com.clust4j.utils.parallel.map;

import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class DistributedMatrixMultiplication extends DualMatrixMapTaskOperator {
	private static final long serialVersionUID = 6656335357116703359L;
	final int a_row, b_col;

	private DistributedMatrixMultiplication(double[][] mat, 
			double[][] mat_b, double[][] result, int lo, int hi) {
		super(mat, mat_b, result, lo, hi);
		a_row = mat.length;
		b_col = mat_b.length; // It's transposed
	}

	@Override
	protected double[][] operate(int lo, int hi) {
		for(int i = lo; i < hi; i++)
			for(int j = 0; j < b_col; j++)
				matrix_c[i][j] = VecUtils
					.innerProductForceSerial // Should force serial?
						(matrix[i], matrix_b[j]);
		return matrix_c; // Unnecessary in this context (using mutability) except for erasure
	}

	@Override
	protected DistributedMatrixMultiplication newInstance(double[][] a, 
			double[][] b, double[][] c, int low, int high) {
		return new DistributedMatrixMultiplication(a,b,c,low,high);
	}
	
	public static double[][] operate(final double[][] a, final double[][] b) {
		MatUtils.checkMultipliability(a,b);
		final double[][] c = new double[a.length][b[0].length];
		return getThreadPool().invoke(new DistributedMatrixMultiplication(a,MatUtils.transpose(b),c,0,a.length));
    }
}
