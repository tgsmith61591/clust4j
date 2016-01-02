package com.clust4j.utils.parallel;

import static com.clust4j.utils.parallel.ConcurrencyUtils.MAX_DIST_LEN;

import java.util.concurrent.RecursiveTask;

abstract class DistributedVectorTask<T> extends RecursiveTask<T> {
	private static final long serialVersionUID = -7986981765361158408L;
	public static final int MAX_CHUNK_SIZE = MAX_DIST_LEN / Runtime.getRuntime().availableProcessors(); //2_500_000;

    public final double[] array;
	public final int low;
    public final int high;
	
	DistributedVectorTask(double[] arr, int lo, int hi) {
		array = arr;
		low = lo;
		high = hi;
	}
}
