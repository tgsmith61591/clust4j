package com.clust4j;

import java.util.concurrent.ForkJoinPool;

/**
 * A set of global values used in multiple classes. Some values may
 * be set to the user's preference, while others are final.
 * 
 * @author Taylor G Smith
 */
public class GlobalState {
	
	/**
	 * The number of available cores on the machine. Used for determining
	 * whether or not to use parallelism & how large parallel chunks should be. */
	public static final int NUM_CORES = Runtime.getRuntime().availableProcessors();
	
	/** If true and the size of the vector exceeds {@value ParallelismUtils#MAX_SERIAL_VECTOR_LEN}, 
	 *  auto schedules parallel job for applicable operations. This can slow
	 *  things down on machines with a lower core count, but speed them up
	 *  on machines with a higher core count. More heap space may be required. 
	 *  Default value is true if availableProcessors is at least 8 */
	public static boolean ALLOW_PARALLELISM = NUM_CORES >= 8;
	
	/**
	 * The global ForkJoin thread pool for parallel recursive tasks. */
	final static public ForkJoinPool FJ_THREADPOOL = new ForkJoinPool();
	
	/**
	 * The max length a vector may be before defaulting to a parallel process, if applicable */
	static public int MAX_SERIAL_VECTOR_LEN = 10_000_000;

	/** 
	 * The max length a parallel-processed chunk may be */
	public static int MAX_PARALLEL_CHUNK_SIZE = MAX_SERIAL_VECTOR_LEN / NUM_CORES; //2_500_000;
}
