package com.clust4j;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;

import com.clust4j.algo.preprocess.FeatureNormalization;

/**
 * A set of global config values used in multiple classes. Some values may
 * be set to the user's preference, while others are final.
 * 
 * @author Taylor G Smith
 */
public class GlobalState {
	/** The default random state */
	public final static Random DEFAULT_RANDOM_STATE = new Random();
	
	
	
	/**
	 * Holds static mathematical values
	 * @author Taylor G Smith
	 */
	public static final class Mathematics {
		/** Double.MIN_VALUE is not negative; this is */
		public final static double SIGNED_MIN = Double.NEGATIVE_INFINITY;
		public final static double SIGNED_MAX = Double.POSITIVE_INFINITY;
		public final static double TINY = 2.2250738585072014e-308;
		public final static double EPS = 2.2204460492503131e-16;
	}
	
	
	
	/**
	 * A class to hold configurations for parallelism
	 * @author Taylor G Smith
	 */
	public final static class ParallelismConf {
		/**
		 * The minimum number of required cores to efficiently
		 * allow parallel operations.
		 */
		public static final int MIN_PARALLEL_CORES_REQUIRED = 8;
		
		/**
		 * The number of available cores on the machine. Used for determining
		 * whether or not to use parallelism & how large parallel chunks should be. */
		public static final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		
		/** If true and the size of the vector exceeds {@value ParallelismUtils#MAX_SERIAL_VECTOR_LEN}, 
		 *  auto schedules parallel job for applicable operations. This can slow
		 *  things down on machines with a lower core count, but speed them up
		 *  on machines with a higher core count. More heap space may be required. 
		 *  Default value is true if availableProcessors is at least 8 */
		public static boolean ALLOW_PARALLELISM = NUM_CORES >= MIN_PARALLEL_CORES_REQUIRED;
		
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
	
	
	
	
	/**
	 * A class to hold configurations for FeatureNormalization
	 * @author Taylor G Smith
	 */
	public final static class FeatureNormalizationConf {
		/** The lower bound for the {@link FeatureNormalization#MIN_MAX_SCALE} scaling range. */
		public static int MIN_MAX_SCALER_RANGE_MIN = 0;
		/** The upper bound for the {@link FeatureNormalization#MIN_MAX_SCALE} scaling range. */
		public static int MIN_MAX_SCALER_RANGE_MAX = 1;
	}
}
