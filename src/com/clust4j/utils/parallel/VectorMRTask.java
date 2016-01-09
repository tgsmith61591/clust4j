package com.clust4j.utils.parallel;

import static com.clust4j.GlobalState.ParallelismConf.MAX_PARALLEL_CHUNK_SIZE;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import com.clust4j.GlobalState;

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
