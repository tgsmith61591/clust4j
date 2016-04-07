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
package com.clust4j.utils.parallel;

import static com.clust4j.GlobalState.ParallelismConf.MAX_PARALLEL_CHUNK_SIZE;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import com.clust4j.GlobalState;

/**
 * Deprecating these vector parallelism classes, as they perform poorly
 * and shouldn't really be used within algorithms that might working in
 * a parallel fashion.
 * @author Taylor G Smith
 * @param <T>
 */
@Deprecated
public abstract class VectorMRTask<T> extends RecursiveTask<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1703077200538377673L;

    final public double[] array;
	final public int low;
    final public int high;
    
    public VectorMRTask(double[] arr, int lo, int hi) {
		array = arr;
		low = lo;
		high = hi;
	}
	
	public static int getChunkSize() {
		return MAX_PARALLEL_CHUNK_SIZE;
	}
	
	public static ForkJoinPool getThreadPool() {
		return GlobalState.ParallelismConf.FJ_THREADPOOL;
	}
}
