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
