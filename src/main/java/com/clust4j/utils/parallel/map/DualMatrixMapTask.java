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
