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

public abstract class MatrixMRTask<T> extends RecursiveTask<T> {
	private static final long serialVersionUID = -4091298557784484895L;

	final public double[][] matrix;
	final public int low;
    final public int high;
    
    public MatrixMRTask(double[][] mat, int lo, int hi) {
		matrix = mat;
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
