/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
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
					.innerProduct(matrix[i], matrix_b[j]);
		return matrix_c; // Unnecessary in this context (using mutability) except for erasure
	}

	/**
	 * Unnecessary for this specific class, but fits type erasure
	 */
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
