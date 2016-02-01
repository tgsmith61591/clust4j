package com.clust4j.algo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.IllegalClusterStateException;
import com.clust4j.utils.Convergeable;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.NoiseyClusterer;
import com.clust4j.utils.VecUtils;

/**
 * Mean shift is a procedure for locating the maxima of a density function given discrete 
 * data sampled from that function. It is useful for detecting the modes of this density. 
 * This is an iterative method, and we start with an initial estimate <i>x</i> . Let a
 * {@link RadialBasisKernel} function be given. This function determines the weight of nearby 
 * points for re-estimation of the mean. Typically a {@link GaussianKernel} kernel on the 
 * distance to the current estimate is used.
 * 
 * @see <a href="https://en.wikipedia.org/wiki/Mean_shift">Mean shift on Wikipedia</a>
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;, adapted from <a href="https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/mean_shift_.py">sklearn implementation</a>
 */
public class MeanShift 
		extends AbstractDensityClusterer 
		implements CentroidLearner, Convergeable, NoiseyClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4423672142693334046L;
	
	final public static double DEF_BANDWIDTH = 5.0;
	final public static int DEF_MAX_ITER = 300;
	final public static double DEF_MIN_CHANGE = 0d;
	final public static int DEF_MIN_BIN_FREQ = 1;
	final public static int NOISE_CLASS = -1;
	
	
	
	
	/** The max iterations */
	private final int maxIter;
	
	/** Min change convergence criteria */
	private final double minChange;
	
	/** The kernel bandwidth */
	private final double bandwidth;

	/** Class labels */
	private volatile int[] labels = null;
	
	/** The M x N seeds to be used as initial kernel points */
	private double[][] seeds;
	
	/** Num rows, cols */
	private final int m, n;

	
	/** Track convergence */
	private volatile boolean converged = false;
	/** The centroid records */
	private volatile ArrayList<double[]> centroids;
	private volatile int numClusters;
	private volatile int numNoisey;
	/** Count iterations */
	private volatile int itersElapsed = 0;
	
	
	
	/**
	 * Default constructor
	 * @param data
	 * @param bandwidth
	 */
	public MeanShift(AbstractRealMatrix data, final double bandwidth) {
		this(data, new MeanShiftPlanner(bandwidth));
	}
	
	/**
	 * Constructor with custom MeanShiftPlanner
	 * @param data
	 * @param planner
	 */
	public MeanShift(AbstractRealMatrix data, MeanShiftPlanner planner) {
		super(data, planner);
		
		
		// Check bandwidth...
		if(planner.bandwidth <= 0.0)
			throw new IllegalArgumentException("bandwidth must be greater than 0.0");
		
		
		// Check seeds dimension
		String e;
		if(null != planner.seeds) {
			if(planner.seeds.length == 0) {
				e = "seeds length must be greater than 0";
				error(e);
				throw new IllegalArgumentException(e);
			}
			
			if(planner.seeds[0].length != (n=data.getColumnDimension())) {
				e = "seeds column dims do not match data column dims";
				error(e);
				throw new DimensionMismatchException(planner.seeds[0].length, n);
			}
			
			info("initializing kernels from given seeds");
			seeds = MatUtils.copy(planner.seeds);
		} else { // Default = all
			info("no seeds provided; defaulting to all datapoints");
			seeds = data.getData();
			n = data.getColumnDimension();
		}
		
		
		
		this.bandwidth = planner.bandwidth;
		this.maxIter = planner.maxIter;
		this.minChange = planner.minChange;
		this.m = data.getRowDimension();
		
		
		/*// No longer need the test...
		this.seeds_m = seeds.length;
		if(seeds_m > m) {
			e = "seeds length cannot exceed number of datapoints";
			if(verbose) error(e);
			throw new IllegalArgumentException(e);
		}
		*/
		
		
		meta("bandwidth="+bandwidth);
		meta("maxIter="+maxIter);
		meta("minChange="+minChange);
	}
	
	
	
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class MeanShiftPlanner extends AbstractClusterer.BaseClustererPlanner {
		private double bandwidth;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_MIN_CHANGE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private double[][] seeds = null;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		
		public MeanShiftPlanner() {
			this(DEF_BANDWIDTH);
		}
		
		public MeanShiftPlanner(final double bandwidth) {
			this.bandwidth = bandwidth;
		}
		

		
		@Override
		public MeanShift buildNewModelInstance(AbstractRealMatrix data) {
			return new MeanShift(data, this);
		}
		
		@Override
		public MeanShiftPlanner copy() {
			return new MeanShiftPlanner(bandwidth)
				.setMaxIter(maxIter)
				.setMinChange(minChange)
				.setScale(scale)
				.setSeed(seed)
				.setSeeds(seeds)
				.setSep(dist)
				.setVerbose(verbose)
				.setNormalizer(norm);
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
		
		public MeanShiftPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}
		
		public MeanShiftPlanner setMinChange(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		public MeanShiftPlanner setSeeds(final double[][] seeds) {
			this.seeds = null == seeds ? seeds : MatUtils.copy(seeds);
			return this;
		}
		
		@Override
		public MeanShiftPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public MeanShiftPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	public double getBandwidth() {
		return bandwidth;
	}
	
	@Override
	public boolean didConverge() {
		return converged;
	}
	
	/**
	 * Returns the max number of iterations 
	 * required for algorithm convergence
	 */
	@Override
	public int itersElapsed() {
		return itersElapsed;
	}
	
	/**
	 * Returns a copy of the seeds matrix
	 * @return
	 */
	public double[][] getKernelSeeds() {
		return MatUtils.copy(seeds);
	}
	
	public int getMaxIter() {
		return maxIter;
	}
	
	public double getConvergenceTolerance() {
		return minChange;
	}

	@Override
	public String getName() {
		return "MeanShift";
	}


	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.MEANSHIFT;
	}


	@Override
	public MeanShift fit() {
		synchronized(this) { // Synch because isTrained is a race condition
			
			try {
				if(null!=labels) // Already fit this model
					return this;
				info("identifying neighborhoods within bandwidth");
				
				
				// Init labels, centroids
				converged = true; // Will reset to false in loop in needed
				final long start = System.currentTimeMillis();
				labels = new int[m];
				centroids = new ArrayList<double[]>();
				String error; // Hold any error msgs
				
				
				// Now get single seed members
				info("computing points within seed bandwidth radii");
				ArrayList<EntryPair<double[], Integer>> all_res = new ArrayList<>();
				for(double[] seed: seeds)
					all_res.add(meanShiftSingleSeed(seed));
				
				
				// Put the results into a Map (hash because tree imposes comparable casting)
				HashMap<double[], Integer> center_intensity = new HashMap<>();
				for(int i = 0; i < m; i++)
					if(null != all_res.get(i))
						center_intensity.put(all_res.get(i).getKey(), all_res.get(i).getValue());
				
				
				// Check for points all too far from seeds
				if(center_intensity.isEmpty()) {
					error = "No point was within bandwidth="+bandwidth+
							" of any seed. Increase the bandwidth or try different seeds.";
					error(error);
					throw new IllegalClusterStateException(error);
				}
				
				
				// Post-processing. Remove near duplicate seeds
				// If dist btwn two kernels is less than bandwidth, remove one w fewer pts
				info("identifying most populated seeds, removing near-duplicates");
				SortedSet<Map.Entry<double[], Integer>> sorted_by_intensity = 
					ClustUtils.sortEntriesByValue(center_intensity, true);
				
				
				// Extract the centroids
				int idx = 0;
				final double[][] sorted_centers = new double[sorted_by_intensity.size()][];
				for(Map.Entry<double[], Integer> entry: sorted_by_intensity)
					sorted_centers[idx++] = entry.getKey();
				
				
				// Create a boolean mask, init true
				final boolean[] unique = new boolean[sorted_centers.length];
				for(int i = 0; i < unique.length; i++) unique[i] = true;
				
				
				
				// Iterate over sorted centers
				int redundant_ct = 0;
				ArrayList<Integer> indcs;
				final double[][] cent_dist_mat = ClustUtils.distanceUpperTriangMatrix(sorted_centers, getSeparabilityMetric());
				for(int i = 0; i < sorted_centers.length; i++) {
					if(unique[i]) {
						indcs = NearestNeighbors.getNearestWithinRadius(bandwidth, cent_dist_mat, i);
						
						for(Integer id: indcs) {
							unique[id] = false;
							redundant_ct++;
						}
						
						unique[i] = true; // Keep this as true
					}
				}
				
				
				// Now assign the centroids...
				for(int i = 0; i < unique.length; i++)
					if(unique[i])
						centroids.add(sorted_centers[i]);
				
				info((numClusters=centroids.size())+" optimal kernels identified");
				info(redundant_ct + " nearly-identical kernel" + 
						(redundant_ct!=1?"s":"") + " removed");
				
				
				// Assign labels now
				final long clustStart = System.currentTimeMillis();
				for(int i = 0; i < unique.length; i++) {
					final double[] record = data.getRow(i);
					
					int closest_cent = NOISE_CLASS;
					double min_dist = Double.MAX_VALUE;
					for(int j = 0; j < centroids.size(); j++) {
						final double[] centroid = centroids.get(j);
						double dist = getSeparabilityMetric().getDistance(record, centroid);
						
						if(dist < min_dist && dist <= bandwidth) {
							closest_cent = j;
							min_dist = dist;
						}
					}
					
					labels[i] = closest_cent;
				}
				
				// Wrap up...
				// Count missing
				numNoisey = 0;
				for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
				info(numNoisey+" record"+(numNoisey!=1?"s":"")+ " classified noise");
				
				
				info("completed cluster labeling in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-clustStart, false));
				
				
				info("model "+getKey()+" completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
					System.lineSeparator());
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synch
		
	} // End train


	@Override
	public ArrayList<double[]> getCentroids() {
		try {
			final ArrayList<double[]> cent = new ArrayList<double[]>();
			for(double[] d : centroids)
				cent.add(VecUtils.copy(d));
			
			return cent;
		} catch(NullPointerException e) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
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
	
	private Integer[] getNeighbors(final double[] seed) {
		final ArrayList<Integer> output = new ArrayList<Integer>();
		for(int i = 0; i < data.getRowDimension(); i++) {
			double dist = getSeparabilityMetric().getDistance(seed, data.getRow(i));
			if(dist < bandwidth) output.add(i);
		}
		
		return output.toArray(new Integer[output.size()]);
	}
	
	private EntryPair<double[], Integer> meanShiftSingleSeed(double[] seed) {

		double norm, diff;
		int completed_iterations = 0;
		
		while(true) {
			// Keep track of max iterations elapsed
			if(completed_iterations > itersElapsed)
				itersElapsed = completed_iterations;
			
			
			final Integer[] i_nbrs = getNeighbors(seed);
			
			// Check if exit
			if(i_nbrs.length == 0) {
				//if(verbose) info("breaking from single seed computation early due to empty neighbor set");
				break;
			}
			
			// Save the old seed
			final double[] oldSeed = seed;
			
			
			// Get the points inside and simultaneously calc new seed
			final double[] newSeed = new double[n];
			for(Integer rec: i_nbrs) {
				final double[] record = data.getRow(rec);
				for(int j = 0; j < n; j++)
					newSeed[j] += record[j];
			}
			
			// Set newSeed to means
			// Also calculate running norm to avoid 2N extra passes
			norm = 0; diff = 0;
			for(int j = 0; j < n; j++) {
				newSeed[j] /= (double) i_nbrs.length;
				diff = newSeed[j] - oldSeed[j];
				norm += diff * diff;
			}
			
			// Assign the new seed
			seed = newSeed;
			norm = FastMath.sqrt(norm);
			
			// Check stopping criteria
			if( norm < minChange || 
					completed_iterations == maxIter )
				return new EntryPair<double[], Integer>(seed, i_nbrs.length);
			completed_iterations++;
		}
		
		converged = false;
		return null; // Default if breaks from inner loop
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
