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
package com.clust4j.utils.parallel.reduce;
import com.clust4j.utils.parallel.VectorMRTask;

@Deprecated
abstract class VectorReduceTask<T> extends VectorMRTask<T> {
	private static final long serialVersionUID = -7986981765361158408L;
	
	VectorReduceTask(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
		checkDims(arr);
	}
    
    /**
     * How to join two values in the result
     */
    protected abstract T joinSides(final T left, final T right);

    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected T operate(final int lo, final int hi);
    
    /**
     * Check dims
     * @param v
     */
    abstract void checkDims(double[] v);
}
