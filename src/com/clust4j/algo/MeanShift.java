package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.RadiusNeighbors.RadiusNeighborsPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.except.IllegalClusterStateException;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
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
	final public static int DEF_MIN_BIN_FREQ = 1;
	final static int maxTries = 8;
	final static double incrementAmt = 0.25;
	
	
	
	
	/** The max iterations */
	private final int maxIter;
	
	/** Min change convergence criteria */
	private final double tolerance;
	
	/** The kernel bandwidth (volatile because can change in sync method) */
	volatile private double bandwidth;

	/** Class labels */
	volatile private int[] labels = null;
	
	/** The M x N seeds to be used as initial kernel points */
	private double[][] seeds;
	
	/** Num rows, cols */
	private final int m, n;
	
	/** Whether bandwidth is auto-estimated */
	private final boolean autoEstimate;

	
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
	 * Default constructor for auto bandwidth estimation
	 * @param data
	 * @param bandwidth
	 */
	public MeanShift(AbstractRealMatrix data) {
		this(data, new MeanShiftPlanner());
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
				throw new org.apache.commons.math3.exception.DimensionMismatchException(planner.seeds[0].length, n);
			}
			
			if(seeds.length > data.getRowDimension()) {
				e = "seeds length cannot exceed number of datapoints";
				error(e);
				throw new IllegalArgumentException(e);
			}
			
			info("initializing kernels from given seeds");
			seeds = MatUtils.copy(planner.seeds);
		} else { // Default = all*/
			info("no seeds provided; defaulting to all datapoints");
			seeds = data.getData();
			n = data.getColumnDimension();
		}
		
		
		this.maxIter = planner.maxIter;
		this.tolerance = planner.minChange;
		this.m = seeds.length; //data.getRowDimension();
		

		this.autoEstimate = planner.autoEstimateBW;
		final LogTimer aeTimer = new LogTimer();
		this.bandwidth = autoEstimate ? 
			autoEstimateBW(this.data, // Needs to be 'this' because might be stdized
				planner.autoEstimateBWQuantile, 
				planner.getSep(), planner.seed) : 
				planner.bandwidth;
			
		if(autoEstimate) info("bandwidth auto-estimated in " + aeTimer.toString());
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Bandwidth","Scale","Force Par.","Allow Par.","Max Iter.","Tolerance"
			}, new Object[]{
				data.getRowDimension(),data.getColumnDimension(),
				getSeparabilityMetric(),
				(autoEstimate ? "(auto) " : "") + bandwidth,
				normalized,
				GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE,
				GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM,
				maxIter, tolerance
			});
	}
	
	final protected static double autoEstimateBW(AbstractRealMatrix data, double quantile, GeometricallySeparable sep, Random seed) {
		if(quantile <= 0 || quantile > 1)
			throw new IllegalArgumentException("illegal quantile");
		final int m = data.getRowDimension(), nnbrs = (int)(m * quantile);
		
		NearestNeighbors nn = new NearestNeighbors(data,
			new NearestNeighborsPlanner(nnbrs)
				.setSeed(seed)).fit();
		
		double bw = 0.0;
		final double[][] X = data.getData();
		final int chunkSize = 500;
		final int numChunks = getNumChunks(chunkSize, m);
		Neighborhood neighb;
		
		
		/*
		 * For each chunk of 500, get the neighbors and then compute the
		 * sum of the row maxes of the distance matrix.
		 */
		int chunkStart, nextChunk;
		for(int chunk = 0; chunk < numChunks; chunk++) {
			chunkStart = chunk * chunkSize;
			nextChunk = chunk == numChunks - 1 ? m : chunkStart + chunkSize;
			
			double[][] nextMatrix = new double[nextChunk - chunkStart][];
			for(int i = chunkStart, j = 0; i < nextChunk; i++, j++)
				nextMatrix[j] = X[i];
			
			neighb = nn.getNeighbors(nextMatrix);
			for(double[] distRow: neighb.getDistances())
				bw += VecUtils.max(distRow);
		}
		
		return bw / (double)m;
	}
	
	protected static int getNumChunks(final int chunkSize, final int m) {
		return (int)FastMath.ceil( ((double)m)/((double)chunkSize) );
	}
	
	
	
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class MeanShiftPlanner 
			extends AbstractClusterer.BaseClustererPlanner 
			implements UnsupervisedClassifierPlanner {
		
		private boolean autoEstimateBW = false;
		private double autoEstimateBWQuantile = 0.3;
		private double bandwidth = DEF_BANDWIDTH;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_TOL;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private double[][] seeds = null;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		
		public MeanShiftPlanner() {
			this.autoEstimateBW = true;
		}
		
		public MeanShiftPlanner(final double bandwidth) {
			this.bandwidth = bandwidth;
		}
		

		
		@Override
		public MeanShift buildNewModelInstance(AbstractRealMatrix data) {
			return new MeanShift(data, this.copy());
		}
		
		@Override
		public MeanShiftPlanner copy() {
			return new MeanShiftPlanner(bandwidth)
				.setAutoBandwidthEstimation(autoEstimateBW)
				.setAutoBandwidthEstimationQuantile(autoEstimateBWQuantile)
				.setMaxIter(maxIter)
				.setMinChange(minChange)
				.setScale(scale)
				.setSeed(seed)
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
		
		public MeanShiftPlanner setAutoBandwidthEstimation(boolean b) {
			this.autoEstimateBW = b;
			return this;
		}
		
		public MeanShiftPlanner setAutoBandwidthEstimationQuantile(double d) {
			this.autoEstimateBWQuantile = d;
			return this;
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
	
	
	/**
	 * Get the kernel bandwidth
	 * @return kernel bandwidth
	 */
	public double getBandwidth() {
		return bandwidth;
	}
	
	/** {@inheritDoc} */
	@Override
	public boolean didConverge() {
		return converged;
	}
	
	/** {@inheritDoc} */
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

	/** {@inheritDoc} */
	@Override
	public int getMaxIter() {
		return maxIter;
	}
	
	/** {@inheritDoc} */
	@Override
	public double getConvergenceTolerance() {
		return tolerance;
	}

	@Override
	public String getName() {
		return "MeanShift";
	}


	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.MEANSHIFT;
	}
	
	static EntryPair<ArrayList<EntryPair<double[], Integer>>, Integer> getCenterIntensity(AbstractRealMatrix data, 
			double bandwidth, double[][] seeds, Random rand, GeometricallySeparable metric, int maxIter) {
		int itrz = 0;
		
		RadiusNeighbors nbrs = new RadiusNeighbors(data,
			new RadiusNeighborsPlanner(bandwidth)
				.setScale(false) // if we scaled in MeanShift, data is already there
				.setSeed(rand)
				.setSep(metric)
				.setVerbose(false)).fit();
		
		// Now get single seed members
		MeanShiftSeed sd;
		TreeSet<MeanShiftSeed> computedSeeds = new TreeSet<>();
		for(double[] seed: seeds) {
			sd = singleSeed(seed, nbrs, seeds, maxIter);
			if(null == sd)
				continue;
			
			computedSeeds.add(sd);
			itrz = FastMath.max(itrz, sd.iterations);
		}
		
		
		ArrayList<EntryPair<double[], Integer>> center_intensity = new ArrayList<>();
		
		// add the entry pairs
		for(MeanShiftSeed seed: computedSeeds) {
			if(null != seed) {
				center_intensity.add(seed.getPair());
			}
		}
		
		return new EntryPair<>(center_intensity, itrz);
	}

	@Override
	public MeanShift fit() {
		synchronized(this) { // Synch because isTrained is a race condition
			
			try {
				if(null!=labels) // Already fit this model
					return this;
				

				int tries = 0;
				final LogTimer timer = new LogTimer();
				

				// Put the results into a Map (hash because tree imposes comparable casting)
				ArrayList<EntryPair<double[], Integer>> center_intensity = null;
				RadiusNeighbors nbrs;
				String error; // Hold any error msgs
				centroids = new ArrayList<double[]>();
				
				int itrz = 0;
				while(true) {
					itrz = 0;

					final EntryPair<ArrayList<EntryPair<double[],Integer>>, Integer> entry =
						getCenterIntensity(data, bandwidth, seeds, getSeed(), 
								getSeparabilityMetric(), maxIter);
					
					center_intensity = entry.getKey();
					itrz = entry.getValue();
					
					// Check for points all too far from seeds
					boolean empty = center_intensity.isEmpty();
					if(empty && tries >= maxTries) {
						error = "No point was within bandwidth="+bandwidth+
								" of any seed. Max tries reached; try increasing bandwidth";
						error(error);
						throw new IllegalClusterStateException(error);
					} else if(empty) {
						error = "No point was within bandwidth="+bandwidth+
								" of any seed. Automatically increasing bandwidth to " + 
								(bandwidth=(bandwidth+incrementAmt)) + " and trying again.";
						warn(error);
						tries++;
					} else {
						converged = true;
						break;
					}
				}
				
				
				
				
				if(tries > 0) // if it was automatically increased
					info("final bandwidth selection: " + bandwidth);
				itersElapsed = itrz; // max iters elapsed
				
				
				
				// Post-processing. Remove near duplicate seeds
				// If dist btwn two kernels is less than bandwidth, remove one w fewer pts
				//info("identifying most populated seeds, removing near-duplicates");
				// Now already sorted desc by value...
				//ArrayList<Map.Entry<double[], Integer>> sorted_by_intensity = 
				//	sortEntriesByValue(center_intensity, true);
				
				final ArrayList<EntryPair<double[], Integer>> sorted_by_intensity = center_intensity;
				
				// Extract the centroids
				int idx = 0, m_prime = sorted_by_intensity.size();
				final Array2DRowRealMatrix sorted_centers = new Array2DRowRealMatrix(m_prime,n);
				for(Map.Entry<double[], Integer> entry: sorted_by_intensity)
					sorted_centers.setRow(idx++, entry.getKey());
				
				
				// Create a boolean mask, init true
				final boolean[] unique = new boolean[m_prime];
				for(int i = 0; i < unique.length; i++) unique[i] = true;
				
				// Fit the new neighbors model
				nbrs = new RadiusNeighbors(sorted_centers,
					new RadiusNeighborsPlanner(bandwidth)
						.setScale(false) // dont scale centers
						.setSeed(getSeed())
						.setSep(getSeparabilityMetric())
						.setVerbose(false)).fit();

				
				// Iterate over sorted centers and query radii
				int redundant_ct = 0;
				int[] indcs;
				double[] center;
				for(int i = 0; i < m_prime; i++) {
					if(unique[i]) {
						center = sorted_centers.getRow(i);
						indcs = nbrs.getNeighbors(
							new double[][]{center}, bandwidth)
								.getIndices()[0];
						
						for(int id: indcs)
							unique[id] = false;
						
						unique[i] = true; // Keep this as true
					}
				}
				
				
				// Now assign the centroids...
				for(int i = 0; i < unique.length; i++) {
					if(unique[i]) {
						centroids.add(sorted_centers.getRow(i));
					}
				}
				
				// calc redundant ct
				redundant_ct = unique.length - centroids.size();
				
				// also put the centroids into a matrix. We have to
				// wait to perform this op, because we have to know
				// the size of centroids first...
				Array2DRowRealMatrix centers = new Array2DRowRealMatrix(centroids.size(),n);
				for(int i = 0; i < centers.getRowDimension(); i++)
					centers.setRow(i, centroids.get(i));
				
				
				// Build yet another neighbors model... 
				// this one has the propensity to throw exceptions
				NearestNeighbors nn; 
				try { // This only happens if centeres.length == 1...
					nn = new NearestNeighbors(centers,
						new NearestNeighborsPlanner(1)
							.setScale(false) // dont scale the centers
							.setSeed(getSeed())
							.setSep(getSeparabilityMetric())
							.setVerbose(false)).fit();
				} catch(IllegalArgumentException iae) {
					error = "only "+centers.getRowDimension()+" centroid"
							+ (centers.getRowDimension()==1?"":"s")+" identified; "
							+ "try altering bandwidth";
					error(error);
					throw new IllegalClusterStateException(error + "; " + iae.getMessage(), iae);
				}
				
				
				info((numClusters=centroids.size())+" optimal kernels identified");
				info(redundant_ct + " nearly-identical kernel" + 
						(redundant_ct!=1?"s":"") + " removed");
				
				
				// Get the nearest...
				final LogTimer clustTimer = new LogTimer();
				Neighborhood knrst = nn.getNeighbors(data);
				labels = MatUtils.flatten(knrst.getIndices());
				
				
				
				// order the labels..
				/* 
				 * Reduce labels to a sorted, gapless, list
				 * sklearn line: cluster_centers_indices = np.unique(labels)
				 */
				ArrayList<Integer> centroidIndices = new ArrayList<Integer>(numClusters);
				for(Integer i: labels) // force autobox
					if(!centroidIndices.contains(i)) // Not race condition because synchronized
						centroidIndices.add(i);
				
				/*
				 * final label assignment...
				 * sklearn line: labels = np.searchsorted(cluster_centers_indices, labels)
				 */
				for(int i = 0; i < labels.length; i++)
					labels[i] = centroidIndices.indexOf(labels[i]);
				
				
				
				// Wrap up...
				// Count missing
				numNoisey = 0;
				for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
				info(numNoisey+" record"+(numNoisey!=1?"s":"")+ " classified noise");
				
				
				info("completed cluster labeling in " + clustTimer.toString());
				
				
				sayBye(timer);
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
	
	static class MeanShiftSeed implements Comparable<MeanShiftSeed> {
		final double[] dists;
		final Integer count;
		final int iterations;
		
		MeanShiftSeed(final double[] dists, final int count, int iterations) {
			this.dists = dists;
			this.count = count;
			this.iterations = iterations;
		}
		
		@Override
		public boolean equals(Object o) {
			if(this == o)
				return true;
			if(o instanceof MeanShiftSeed) {
				MeanShiftSeed m = (MeanShiftSeed)o;
				return VecUtils.equalsExactly(dists, m.dists)
					&& count.intValue() == m.count.intValue();
			}
			
			return false;
		}
		
		@Override
		public int hashCode() {
			int h = 31;
			for(double d: dists)
				h ^= (int)d;
			return h ^ count;
		}
		
		EntryPair<double[],Integer> getPair() {
			return new EntryPair<>(dists, count);
		}
		
		@Override
		public String toString() {
			return "{" + Arrays.toString(dists) + " : " + count + "}";
		}

		@Override
		public int compareTo(MeanShiftSeed o2) {
			int comp = count.compareTo(o2.count);
			
			if(comp == 0) {
				final double[] d2 = o2.dists;
				
				for(int i= 0; i < dists.length; i++) {
					int c = Double.valueOf(dists[i]).compareTo(d2[i]);
					if(c != 0)
						return -c;
				}
			}
			
			return -comp;
		}
	}
	
	static MeanShiftSeed singleSeed(double[] seed, RadiusNeighbors rn, double[][] X, int maxIter) {
		final double bandwidth = rn.getRadius(), tolerance = 1e-3;
		final int n = X[0].length; // we know X is uniform
		int completed_iterations = 0;
		
		double norm, diff;
		
		while(true) {

			Neighborhood nbrs = rn.getNeighbors(new double[][]{seed}, bandwidth);
			int[] i_nbrs = nbrs.getIndices()[0];
			
			// Check if exit
			if(i_nbrs.length == 0) 
				break;
			
			System.out.println("Here");
			// Save the old seed
			final double[] oldSeed = seed;
			
			// Get the points inside and simultaneously calc new seed
			final double[] newSeed = new double[n];
			norm = 0; diff = 0;
			for(int i = 0; i < i_nbrs.length; i++) {
				final double[] record = X[i_nbrs[i]];
				
				for(int j = 0; j < n; j++) {
					newSeed[j] += record[j];
				
					// Last iter hack, go ahead and compute means simultaneously
					if(i == i_nbrs.length - 1) {
						newSeed[j] /= (double) i_nbrs.length;
						diff = newSeed[j] - oldSeed[j];
						norm += diff * diff;
					}
				}
			}
			
			// Assign the new seed
			seed = newSeed;
			norm = FastMath.sqrt(norm);
			
			// Check stopping criteria
			if( norm < tolerance || completed_iterations == maxIter )
				return new MeanShiftSeed(seed, i_nbrs.length, completed_iterations);
			completed_iterations++;
		}
		
		// Default... shouldn't get here though
		return null;
	}
	
	

	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","Num. Clusters","Num. Noise"
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
