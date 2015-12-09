package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.NearestNeighbors.RunMode;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.NoiseyClusterer;
import com.clust4j.utils.VecUtils;


/**
 * <a href="https://en.wikipedia.org/wiki/DBSCAN">DBSCAN</a> (Density Based Spatial Clustering
 * for Applications with Noise) is a data clustering algorithm proposed by Martin Ester, 
 * Hans-Peter Kriegel, Jorg Sander and Xiaowei Xu in 1996. It is a density-based clustering 
 * algorithm: given a set of points in some space, it groups together points that are 
 * closely packed together (points with many nearby neighbors), marking as outliers 
 * points that lie alone in low-density regions (whose nearest neighbors are too far away).
 * 
 * @see <a href="http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf">DBSCAN, 
 * A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise</a>
 * @see {@link AbstractDensityClusterer}
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;, adapted from sklearn implementation by Lars Buitinck
 *
 */
public class DBSCAN extends AbstractDensityClusterer implements NoiseyClusterer {
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
	private volatile int numNoisey;
	
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
		
		
		// Error handle...
		String e;
		if(this.eps <= 0.0) {
			e="eps must be greater than 0.0";
			if(verbose) error(e);
			throw new IllegalArgumentException(e);
		}
		
		if(this.minPts < 1) {
			e="minPts must be greater than 0";
			if(verbose) error(e);
			throw new IllegalArgumentException(e);
		}
	}
	

	
	public double getEps() {
		return eps;
	}
	
	@Override
	public int[] getLabels() {
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
	final public DBSCAN fit() {
		synchronized(this) { // synch because alters internal labels and structs
			
			if(null!=labels) // Then we've already fit this...
				return this;
			
			
			// First get the dist matrix
			final long start = System.currentTimeMillis();
			dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
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
			
			
			
			// Super verbose tracing but only for smaller matrices
			if(verbose && m < 10)
				trace("distance matrix:"+
					new MatrixFormatter()
						.format(new Array2DRowRealMatrix(dist_mat, false)));
			
			
			
			// Do the neighborhood assignments, get sample weights, find core samples..
			final long neighbStart = System.currentTimeMillis();
			labels = new int[m]; // Initialize labels...
			sampleWeights = new double[m]; // Init sample weights...
			coreSamples = new boolean[m];
			
			
			// Fit the nearest neighbor model...
			final NearestNeighbors nnModel = new NearestNeighbors(data, 
				new NearestNeighborsPlanner(RunMode.RADIUS)
					.setRadius(eps)
					.setDistanceMatrix(dist_mat)
					.setScale(false) // Don't need to because if scaled in DBSCAN, data already scaled
					.setSeed(getSeed())
					.setSep(getSeparabilityMetric())
					.setVerbose(false)) // Don't want nested verbosity logging...
				.fit();
			final ArrayList<Integer>[] nearest = nnModel.getNearest();
			
			
			
			ArrayList<Integer> ptNeighbs;
			ArrayList<ArrayList<Integer>> neighborhoods = new ArrayList<>();
			for(int i = 0; i < m; i++) {
				// Each label inits to -1 as noise
				labels[i] = NOISE_CLASS;
				ptNeighbs = nearest[i];
				
				// Add neighborhood...
				int pts;
				neighborhoods.add(ptNeighbs);
				sampleWeights[i] = pts = ptNeighbs.size();
				coreSamples[i] = pts >= minPts;
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
			int nextLabel = 0, v;
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
							
							for(i = 0; i < neighb.size(); i++) {
								v = neighb.get(i);
								if(labels[v] == NOISE_CLASS)
									stack.push(v);
							}
						}
					}
					
					//System.out.println(stack);
					if(stack.size() == 0) {
						if(verbose) info("completed stack for clusterLabel " + nextLabel);
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
				
				
				// Count missing
				numNoisey = 0;
				for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
				
				
				info((numClusters=nextLabel)+" cluster"+(nextLabel!=1?"s":"")+
					" identified, "+numNoisey+" record"+(numNoisey!=1?"s":"")+
						" classified noise");
				
				info("model "+getKey()+" completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
					System.lineSeparator());
			}
			
			
			return this;
		} // End synch
		
	}// End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.DBSCAN;
	}
	
	@Override
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}
	
	@Override
	public int getNumberOfNoisePoints() {
		return numNoisey;
	}
}
