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

abstract class DualMatrixMapTaskOperator extends DualMatrixMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2401423511466814014L;

	DualMatrixMapTaskOperator(double[][] mat, double[][] mat_b, double[][] result, int lo, int hi) {
		super(mat, mat_b, result, lo, hi);
	}

	@Override
    protected double[][] compute() {
        if(high - low <= getChunkSize()) {
            return operate(low, high);
        } else {
            int mid = low + (high - low) / 2;
            DualMatrixMapTaskOperator left  = newInstance(matrix, matrix_b, matrix_c, low, mid);
            DualMatrixMapTaskOperator right = newInstance(matrix, matrix_b, matrix_c, mid, high);
            left.fork();
            right.compute();
            left.join();
            
            return matrix_c;
        }
    }
	
	/**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected double[][] operate(final int lo, final int hi);
    
    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @param c
     * @param low
     * @param high
     * @return
     */
    abstract protected DualMatrixMapTaskOperator newInstance(final double[][] a, final double[][] b, final double[][] c, final int low, final int high);
}
