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

/**
 * A package private class to handle all dual-vector operator
 * mapping tasks. The {@link #operate(double, double)} method must
 * be overridden or a {@link UnsupportedOperationException} will be
 * thrown.
 * @author Taylor G Smith
 */
@Deprecated
abstract class DualVectorMapTaskOperator extends DualVectorMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6003468790911249385L;

	DualVectorMapTaskOperator(double[] arr, double[] arr_b, double[] result, int lo, int hi) {
		super(arr, arr_b, result, lo, hi);
	}

    @Override
    protected double[] compute() {
        if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i) 
                array_c[i] = operate(array[i], array_b[i]);
            return array_c;
         } else {
            int mid = low + (high - low) / 2;
            DualVectorMapTaskOperator left  = newInstance(array, array_b, array_c, low, mid);
            DualVectorMapTaskOperator right = newInstance(array, array_b, array_c, mid, high);
            left.fork();
            right.compute();
            left.join();
            return array;
         }
    }

    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected double operate(final double a, final double b);
    
    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @param c
     * @param low
     * @param high
     * @return
     */
    abstract protected DualVectorMapTaskOperator newInstance(final double[] a, final double[] b, final double[] c, final int low, final int high);
}
