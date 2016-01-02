package com.clust4j.utils.parallel;

import java.util.concurrent.ForkJoinPool;

public class ConcurrencyUtils {
	/**
	 * Global ForkJoin thread pool
	 */
	final static public ForkJoinPool fjPool = new ForkJoinPool();
	final static public int MAX_DIST_LEN = 10_000_000;
}
