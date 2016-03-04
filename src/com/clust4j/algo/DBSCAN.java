package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.algo.RadiusNeighbors.RadiusNeighborsPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
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
public class DBSCAN extends AbstractDBSCAN {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6749407933012974992L;
	final private int m;
	
	// Race conditions exist in retrieving either one of these...
	private volatile int[] labels = null;
	private volatile double[] sampleWeights = null;
	private volatile boolean[] coreSamples = null;
	private volatile int numClusters;
	private volatile int numNoisey;
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class DBSCANPlanner extends AbstractDBSCANPlanner {
		private double eps = DEF_EPS;
		private int minPts = DEF_MIN_PTS;
		private boolean scale = DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private Random seed = DEF_SEED;
		private FeatureNormalization norm = DEF_NORMALIZER;
		
		
		public DBSCANPlanner() { }
		public DBSCANPlanner(final double eps) {
			this.eps = eps;
		}

		
		@Override
		public DBSCAN buildNewModelInstance(AbstractRealMatrix data) {
			return new DBSCAN(data, this.copy());
		}
		
		@Override
		public DBSCANPlanner copy() {
			return new DBSCANPlanner(eps)
				.setMinPts(minPts)
				.setScale(scale)
				.setSep(dist)
				.setSeed(seed)
				.setVerbose(verbose)
				.setNormalizer(norm);
		}

		@Override
		public int getMinPts() {
			return minPts;
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
		
		@Override
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
		
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}
		
		@Override
		public DBSCANPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	/**
	 * Constructs an instance of DBSCAN from the default epsilon
	 * @param data
	 */
	public DBSCAN(final AbstractRealMatrix data) {
		this(data, DEF_EPS);
	}
	
	
	/**
	 * Constructs an instance of DBSCAN from the default planner values
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
	public DBSCAN(final AbstractRealMatrix data, final DBSCANPlanner planner) {
		super(data, planner);
		m = data.getRowDimension();
		
		// Error handle...
		String e;
		if(this.eps <= 0.0) {
			e="eps must be greater than 0.0";
			error(e);
			throw new IllegalArgumentException(e);
		}
		
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Epsilon","Min Pts.","Scale","Force Par.","Allow Par."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),
				eps, minPts, normalized,
				GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE,
				GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM
			});
	}
	

	
	public double getEps() {
		return eps;
	}
	
	@Override
	public int[] getLabels() {
		try {
			return VecUtils.copy(labels);
		} catch(NullPointerException npe) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
		}
	}
	
	@Override
	public String getName() {
		return "DBSCAN";
	}
	
	@Override
	final public DBSCAN fit() {
		synchronized(this) { // synch because alters internal labels and structs
			
			try {
				if(null!=labels) // Then we've already fit this...
					return this;
				
				
				// First get the dist matrix
				final LogTimer timer = new LogTimer();
				
				// Do the neighborhood assignments, get sample weights, find core samples..
				final LogTimer neighbTimer = new LogTimer();
				labels = new int[m]; // Initialize labels...
				sampleWeights = new double[m]; // Init sample weights...
				coreSamples = new boolean[m];
				
				
				// Fit the nearest neighbor model...
				final LogTimer rnTimer = new LogTimer();
				final RadiusNeighbors rnModel = new RadiusNeighbors(data,
					new RadiusNeighborsPlanner(eps)
						.setScale(false) // Don't need to because if scaled in DBSCAN, data already scaled
						.setSeed(getSeed())
						.setSep(getSeparabilityMetric())
						.setNormalizer(normer) // Don't really need because not normalizing...
						.setVerbose(false))
					.fit();
				
				info("fit RadiusNeighbors model in " + rnTimer.toString());
				int[][] nearest = rnModel.getNeighbors().getIndices();
				
				
				int[] ptNeighbs;
				ArrayList<int[]> neighborhoods = new ArrayList<>();
				int numCorePts = 0;
				for(int i = 0; i < m; i++) {
					// Each label inits to -1 as noise
					labels[i] = NOISE_CLASS;
					ptNeighbs = nearest[i];
					
					// Add neighborhood...
					int pts;
					neighborhoods.add(ptNeighbs);
					sampleWeights[i] = pts = ptNeighbs.length;
					coreSamples[i] = pts >= minPts;
					
					if(coreSamples[i]) 
						numCorePts++;
				}
				
				
				// Log checkpoint
				info("completed density neighborhood calculations in " + neighbTimer.toString());
				info(numCorePts + " core point"+(numCorePts!=1?"s":"")+" found");
				
				
				// Label the points...
				int nextLabel = 0, v;
				final Stack<Integer> stack = new Stack<>();
				int[] neighb;
				
				
				LogTimer stackTimer = new LogTimer();
				for(int i = 0; i < m; i++) {
					stackTimer = new LogTimer();
					
					// Want to look at unlabeled OR core points...
					if(labels[i] != NOISE_CLASS || !coreSamples[i])
						continue;
					
			        // Depth-first search starting from i, ending at the non-core points.
			        // This is very similar to the classic algorithm for computing connected
			        // components, the difference being that we label non-core points as
			        // part of a cluster (component), but don't expand their neighborhoods.
					int labelCt = 0;
					while(true) {
						if(labels[i] == NOISE_CLASS) {
							labels[i] = nextLabel;
							labelCt++;
							
							if(coreSamples[i]) {
								neighb = neighborhoods.get(i);
								
								for(i = 0; i < neighb.length; i++) {
									v = neighb[i];
									if(labels[v] == NOISE_CLASS)
										stack.push(v);
								}
							}
						}
						

						if(stack.size() == 0) {
							fitSummary.add(new Object[]{
								nextLabel, labelCt, stackTimer.formatTime(), stackTimer.wallTime()
							});
							
							break;
						}
						
						i = stack.pop();
					}
					
					nextLabel++;
				}
				
				
				// Count missing
				numNoisey = 0;
				for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
				
				
				// corner case: numNoisey == m (never gets a fit summary)
				if(numNoisey == m)
					fitSummary.add(new Object[]{
						Double.NaN, 0, stackTimer.formatTime(), stackTimer.wallTime()
					});
				
				
				
				info((numClusters=nextLabel)+" cluster"+(nextLabel!=1?"s":"")+
					" identified, "+numNoisey+" record"+(numNoisey!=1?"s":"")+
						" classified noise");
				
				sayBye(timer);
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
			
		} // End synch
		
	}// End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.DBSCAN;
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Cluster #","Num. Core Pts.","Iter. Time","Wall"
		};
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
