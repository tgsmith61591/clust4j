package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.VecUtils;


/**
 * 
 * 
 * 
 * @author Taylor G Smith, adapted from sklearn implementation by Lars Buitinck
 *
 */
public class DBSCAN extends AbstractDensityClusterer implements Classifier {
	final public static double DEF_EPS = 0.5;
	final public static int DEF_MIN_PTS = 5;
	final public static int NOISE_CLASS = -1;
	
	final private int minPts;
	final private double eps;
	
	
	// Race conditions exist in retrieving either one of these...
	private volatile int[] labels = null;
	private volatile double[] sampleWeights = null;
	private volatile boolean[] coreSamples = null;
	private volatile int numClusters;
	
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
		private double eps = DEF_EPS;
		private int minPts = DEF_MIN_PTS;
		private boolean scale = DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private Random seed = DEF_SEED;
		
		
		public DBSCANPlanner() { }
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
		
		
		if(verbose) {
			meta("epsilon="+builder.eps);
			meta("min_pts="+builder.minPts);
		}
		
		
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
	public int predict(final double[] newRecord) {
		// TODO:
		return 0;
	}
	
	@Override
	final public DBSCAN fit() {
		synchronized(this) { // synch because alters internal labels and structs
			
			
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
				
				info("computing density neighborhood for each point (eps=" + eps + ")");
			}
			
			
			// Do the neighborhood assignments, get sample weights, find core samples..
			final long neighbStart = System.currentTimeMillis();
			labels = new int[m]; // Initialize labels...
			sampleWeights = new double[m]; // Init sample weights...
			coreSamples = new boolean[m];
			
			
			ArrayList<Integer> ptNeighbs;
			ArrayList<ArrayList<Integer>> neighborhoods = new ArrayList<>();
			for(int i = 0; i < m; i++) {
				// Each label inits to -1 as noise
				labels[i] = NOISE_CLASS;
				
				try {
					ptNeighbs = new NearestNeighbor(i, dist_mat).getNearestWithinRadius(eps);
				} catch(DimensionMismatchException e) {
					// Should not happen since i < m
					if(verbose) error(e.getLocalizedMessage());
					throw new InternalError(i+", "+m);
				}
				
				// Add neighborhood...
				int pts;
				neighborhoods.add(ptNeighbs);
				sampleWeights[i] = pts = ptNeighbs.size();
				coreSamples[i] = pts > minPts;
			}
			
			
			// Log checkpoint
			if(verbose) {
				final int numCorePts = VecUtils.sum(coreSamples);
				info("completed density neighborhood calculations in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-neighbStart, false));
				info(numCorePts + " core point"+(numCorePts!=1?"s":"")+" found");
				info("identifying cluster labels");
			}
			
			
			// Label the points...
			int nextLabel = 0, j, v;
			final long clustStart = System.currentTimeMillis();
			final Stack<Integer> stack = new Stack<>();
			ArrayList<Integer> neighb;
			
			
			for(int i = 0; i < m; i++) {
				// Want to look at unlabeled OR core points...
				if(labels[i] != NOISE_CLASS || !coreSamples[i])
					continue;
				
		        // Depth-first search starting from i, ending at the non-core points.
		        // This is very similar to the classic algorithm for computing connected
		        // components, the difference being that we label non-core points as
		        // part of a cluster (component), but don't expand their neighborhoods.
				while(true) {
					if(labels[i] == -1) {
						labels[i] = nextLabel;
						if(coreSamples[i]) {
							neighb = neighborhoods.get(i);
							
							for(j = 0; j < neighb.size(); j++) {
								v = neighb.get(j);
								if(labels[v] == NOISE_CLASS)
									stack.push(v);
							}
						}
					}
					
					if(stack.size() == 0) {
						if(verbose) info("emptied stack for clusterLabel " + nextLabel);
						break;
					}
					
					i = stack.pop();
				}
				
				nextLabel++;
			}
			
			
			
			// Wrap up...
			if(verbose) {
				info("completed cluster labeling in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-clustStart, false));
				
				info((numClusters=nextLabel)+" label"+(nextLabel!=1?"s":"")+" identified");
				
				info("model "+getKey()+" completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false));
			}
			
			
			return this;
		} // End synch
		
	}// End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.DBSCAN;
	}
	
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}
}
