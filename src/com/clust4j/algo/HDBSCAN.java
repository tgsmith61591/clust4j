package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.HierarchicalClusterer.Linkage;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Hierarchical;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

/**
 * Hierarchical Density-Based Spatial Clustering of Applications with Noise. 
 * Performs DBSCAN over varying epsilon values and integrates the result to 
 * find a clustering that gives the best stability over epsilon. This allows 
 * HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more 
 * robust to parameter selection.
 * 
 * @author Taylor G Smith, adapted from the Python 
 * <a href="https://github.com/lmcinnes/hdbscan">HDBSCAN package</a>, inspired by
 * <a href="http://dl.acm.org/citation.cfm?id=2733381">the paper</a> by 
 * R. Campello, D. Moulavi, and J. Sander
 */
public class HDBSCAN extends AbstractDBSCAN implements Hierarchical {
	private static final long serialVersionUID = -5112901322434131541L;

	final private Linkage linkage;
	
	private volatile int[] labels = null;
	private volatile int numClusters;
	private volatile int numNoisey;
	
	/**
	 * Constructs an instance of DBSCAN from the default values
	 * @param eps
	 * @param data
	 */
	public HDBSCAN(final AbstractRealMatrix data, final int minPts) {
		this(data, new HDBSCANPlanner(minPts));
	}
	
	/**
	 * Constructs an instance of DBSCAN from the provided builder
	 * @param builder
	 * @param data
	 */
	public HDBSCAN(final AbstractRealMatrix data, final HDBSCANPlanner planner) {
		super(data, planner);
		this.linkage = planner.linkage;
		meta("min_pts="+planner.minPts);
		meta("linkage="+linkage);
		HierarchicalClusterer.checkLinkage(this, linkage);
	}
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class HDBSCANPlanner extends AbstractDBSCANPlanner implements Hierarchical {
		private int minPts = DEF_MIN_PTS;
		private boolean scale = DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private Linkage linkage;
		private boolean verbose	= DEF_VERBOSE;
		private Random seed = DEF_SEED;
		private FeatureNormalization norm = DEF_NORMALIZER;
		
		
		public HDBSCANPlanner() { }
		public HDBSCANPlanner(final int minPts) {
			this.minPts = minPts;
		}

		
		@Override
		public HDBSCAN buildNewModelInstance(AbstractRealMatrix data) {
			return new HDBSCAN(data, this);
		}
		
		@Override
		public HDBSCANPlanner copy() {
			return new HDBSCANPlanner(minPts)
				.setScale(scale)
				.setSep(dist)
				.setSeed(seed)
				.setVerbose(verbose)
				.setNormalizer(norm);
		}
		
		@Override
		public Linkage getLinkage() {
			return linkage;
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
		
		public HDBSCANPlanner setLinkage(final Linkage link) {
			this.linkage = link;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setMinPts(final int minPts) {
			this.minPts = minPts;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public HDBSCANPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}
		@Override
		public HDBSCANPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}



	@Override
	public AbstractClusterer fit() {
		synchronized(this) {
			
			try {
				if(null!=labels) // Then we've already fit this...
					return this;
				
				//TODO
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
		}
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
	public Linkage getLinkage() {
		return linkage;
	}

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.HDBSCAN;
	}

	@Override
	public String getName() {
		return "HDBSCAN";
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
