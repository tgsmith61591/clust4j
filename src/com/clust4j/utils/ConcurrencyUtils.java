package com.clust4j.utils;

import java.util.concurrent.ForkJoinPool;

public class ConcurrencyUtils {
	/**
	 * Global ForkJoin thread pool
	 */
	final static ForkJoinPool fjPool = new ForkJoinPool();
}
