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

import org.apache.commons.math3.exception.DimensionMismatchException;

@Deprecated
abstract class DualVectorReduceTask<T> extends VectorReduceTask<T> {
	private static final long serialVersionUID = -4647929958194428774L;
	
	final double[] array_b;
	
	DualVectorReduceTask(double[] arr, double[] arr_b, int lo, int hi) {
		super(arr, lo, hi);
		checkDims(arr_b); // super class handles arr
		
		if(arr.length != arr_b.length)
			throw new DimensionMismatchException(arr.length, arr_b.length);
		array_b = arr_b;
	}

}
