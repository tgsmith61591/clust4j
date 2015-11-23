package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Classifier;


public class DBSCAN extends AbstractDensityClusterer implements Classifier {
	
	final public static int DEF_MIN_PTS = 5;
	
	final private int minPts;
	final private double eps;
	
	private boolean isTrained = false;
	private int[] labels = null;
	
	/**
	 * Upper triangular, M x M matrix denoting distances between records.
	 * Is only populated during training phase and then set to null for 
	 * garbage collection, as a large-M matrix has a high space footprint: O(N^2).
	 * This is only needed during training and then can safely be collected
	 * to free up heap space.
	 */
	private double[][] dist_mat = null;
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class DBSCANPlanner extends AbstractClusterer.BaseClustererPlanner {
		private double eps;
		private int minPts = DEF_MIN_PTS;
		private boolean scale = DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private Random seed = DEF_SEED;
		
		public DBSCANPlanner(final double eps) {
			this.eps = eps;
		}
		
		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public DBSCANPlanner setMinPts(final int minPts) {
			this.minPts = minPts;
			return this;
		}
		
		@Override
		public DBSCANPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public DBSCANPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public DBSCANPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public DBSCANPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
	}
	
	
	
	/**
	 * Constructs an instance of DBSCAN from the default values
	 * @param eps
	 * @param data
	 */
	public DBSCAN(final AbstractRealMatrix data, final double eps) {
		this(data, new DBSCANPlanner(eps));
	}
	
	/**
	 * Constructs an instance of DBSCAN from the provided builder
	 * @param builder
	 * @param data
	 */
	public DBSCAN(final AbstractRealMatrix data, final DBSCANPlanner builder) {
		super(data, builder);
		
		this.minPts = builder.minPts;
		this.eps 	= builder.eps;
		
		if(this.eps <= 0.0)
			throw new IllegalArgumentException("eps must be greater than 0.0");
	}
	

	
	public double getEps() {
		return eps;
	}
	
	@Override
	public int[] getPredictedLabels() {
		return labels;
	}
	
	public int getMinPts() {
		return minPts;
	}
	
	@Override
	public String getName() {
		return "DBSCAN";
	}
	
	@Override
	public boolean isTrained() {
		return isTrained;
	}
	
	@Override
	public int predict(final double[] newRecord) {
		// TODO:
		return 0;
	}
	
	@Override
	final public void train() {
		synchronized(this) { // synch because `isTrained is a race condition
			if(isTrained)
				return;
			
			
			// First get the dist matrix
			final long start = System.currentTimeMillis();
			dist_mat = ClustUtils.distanceMatrix(data, getSeparabilityMetric());
			final int m = dist_mat.length;
			
			
			// Log info...
			if(verbose) {
				info("calculated " + 
					dist_mat.length + " x " + 
					dist_mat.length + 
					" distance matrix in " + 
					LogTimeFormatter.millis( System.currentTimeMillis()-start , false));
				
				info("computing density neighborhood for each point");
			}
			
			
			// Initialize the neighborhood array
			final int[] neighborhoods = new int[m]; // The number of pts in each record's neighborhood
			final boolean[][] mask_mat = new boolean[m][m]; // Mask matrix
			for(int i = 0; i < m-1; i++) {
				for(int j = i + 1; j < m; j++) {
					if(i == j)
						continue;
					
					final boolean res = dist_mat[i][j] < eps;
					mask_mat[i][j] = res;
					mask_mat[j][i] = res;
				}
			}
			
			/*
			 * I.e., if EPS == 2...
			 * 
			 * [0 1 2 3 4]       [false TRUE  false false false]
			 * [0 0 1 2 3]       [TRUE  false TRUE  false false]
			 * [0 0 0 1 2]  ===> [false TRUE  false TRUE  false]
			 * [0 0 0 0 1]       [false false TRUE  false TRUE ]
			 * [0 0 0 0 0]       [false false false TRUE  false]
			 */
			
			
			
			
			
			
			// TODO:
			throw new UnsupportedOperationException("Not yet implemented");
		} // End synch
		
	}// End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.DBSCAN;
	}
}
