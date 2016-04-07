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

import com.clust4j.utils.VecUtils;

@Deprecated
abstract class MapTaskOperator extends VectorMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8398439421996975256L;

	MapTaskOperator(double[] arr, int lo, int hi) {
		super(VecUtils.copy(arr), lo, hi);
	}

    @Override
    protected double[] compute() {
        if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i) 
                array[i] = operate(array[i]);
            return array;
         } else {
            int mid = low + (high - low) / 2;
            MapTaskOperator left  = newInstance(array, low, mid);
            MapTaskOperator right = newInstance(array, mid, high);
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
	abstract protected double operate(final double a);
	
	/**
     * Must be overridden by subclasses
     * @param array
     * @param low
     * @param high
     * @return
     */
	abstract protected MapTaskOperator newInstance(final double[] array, final int low, final int high);
}
