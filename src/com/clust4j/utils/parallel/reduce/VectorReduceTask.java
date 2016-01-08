package com.clust4j.utils.parallel.reduce;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import com.clust4j.GlobalState;
import static com.clust4j.GlobalState.ParallelismConf.MAX_PARALLEL_CHUNK_SIZE;

abstract class VectorReduceTask<T> extends RecursiveTask<T> {
	private static final long serialVersionUID = -7986981765361158408L;

    final double[] array;
	final int low;
    final int high;
	
	VectorReduceTask(double[] arr, int lo, int hi) {
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
