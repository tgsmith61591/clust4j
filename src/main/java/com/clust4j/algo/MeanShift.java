/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.algo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.NearestNeighborsParameters;
import com.clust4j.algo.Neighborhood;
import com.clust4j.algo.RadiusNeighborsParameters;
import com.clust4j.except.IllegalClusterStateException;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.SimilarityMetric;
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
final public class MeanShift 
		extends AbstractDensityClusterer 
		implements CentroidLearner, Convergeable, NoiseyClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4423672142693334046L;
	
	final public static double DEF_BANDWIDTH = 5.0;
	final public static int DEF_MAX_ITER = 300;
	final public static int DEF_MIN_BIN_FREQ = 1;
	final static double incrementAmt = 0.25;
	final public static HashSet<Class<? extends GeometricallySeparable>> UNSUPPORTED_METRICS;
	
	
	/**
	 * Static initializer
	 */
	static {
		UNSUPPORTED_METRICS = new HashSet<>();
		// Add metrics here if necessary... already vetoes any
		// similarity metrics, so this might be sufficient...
	}
	
	@Override final public boolean isValidMetric(GeometricallySeparable geo) {
		return !UNSUPPORTED_METRICS.contains(geo.getClass()) && !(geo instanceof SimilarityMetric);
	}
	
	
	
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
	private final int n;
	
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
	protected MeanShift(RealMatrix data, final double bandwidth) {
		this(data, new MeanShiftParameters(bandwidth));
	}
	
	/**
	 * Default constructor for auto bandwidth estimation
	 * @param data
	 * @param bandwidth
	 */
	protected MeanShift(RealMatrix data) {
		this(data, new MeanShiftParameters());
	}
	
	/**
	 * Constructor with custom MeanShiftPlanner
	 * @param data
	 * @param planner
	 */
	protected MeanShift(RealMatrix data, MeanShiftParameters planner) {
		super(data, planner);
		
		
		// Check bandwidth...
		if(planner.getBandwidth() <= 0.0)
			error(new IllegalArgumentException("bandwidth "
				+ "must be greater than 0.0"));
		
		
		// Check seeds dimension
		if(null != planner.getSeeds()) {
			if(planner.getSeeds().length == 0)
				error(new IllegalArgumentException("seeds "
					+ "length must be greater than 0"));
			
			// Throws NonUniformMatrixException if non uniform...
			MatUtils.checkDimsForUniformity(planner.getSeeds());
			
			if(planner.getSeeds()[0].length != (n=this.data.getColumnDimension()))
				error(new DimensionMismatchException(planner.getSeeds()[0].length, n));
			
			if(planner.getSeeds().length > this.data.getRowDimension())
				error(new IllegalArgumentException("seeds "
					+ "length cannot exceed number of datapoints"));
			
			info("initializing kernels from given seeds");
			
			// Handle the copying in the planner
			seeds = planner.getSeeds();
		} else { // Default = all*/
			info("no seeds provided; defaulting to all datapoints");
			seeds = this.data.getData(); // use THIS as it's already scaled...
			n = this.data.getColumnDimension();
		}
		
		/*
		 * Check metric for validity
		 */
		if(!isValidMetric(this.dist_metric)) {
			warn(this.dist_metric.getName() + " is not valid for "+getName()+". "
				+ "Falling back to default Euclidean dist");
			setSeparabilityMetric(DEF_DIST);
		}
		
		
		this.maxIter = planner.getMaxIter();
		this.tolerance = planner.getConvergenceTolerance();
		

		this.autoEstimate = planner.getAutoEstimate();
		final LogTimer aeTimer = new LogTimer();
		
		
		/*
		 * Assign bandwidth
		 */
		this.bandwidth = 
			/* if all singular, just pick a number... */
			this.singular_value ? 0.5 :
			/* Otherwise if we're auto-estimating, estimate it */
			autoEstimate ? 
				autoEstimateBW(this, planner.getAutoEstimationQuantile()) : 
					planner.getBandwidth();
			
		/*
		 * Give auto-estimation timer update	
		 */
		if(autoEstimate && !this.singular_value) info("bandwidth auto-estimated in " + 
			(parallel?"parallel in ":"") + aeTimer.toString());
		
		
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Bandwidth","Allow Par.","Max Iter.","Tolerance"
			}, new Object[]{
				data.getRowDimension(),data.getColumnDimension(),
				getSeparabilityMetric(),
				(autoEstimate ? "(auto) " : "") + bandwidth,
				parallel,
				maxIter, tolerance
			});
	}

	/**
	 * For testing...
	 * @param data
	 * @param quantile
	 * @param sep
	 * @param seed
	 * @param parallel
	 * @return
	 */
	final protected static double autoEstimateBW(Array2DRowRealMatrix data, 
			double quantile, GeometricallySeparable sep, Random seed, boolean parallel) {
		
		return autoEstimateBW(new NearestNeighbors(data,
			new NearestNeighborsParameters((int)(data.getRowDimension() * quantile))
				.setSeed(seed)
				.setForceParallel(parallel)).fit(), 
			data.getDataRef(), 
			quantile, 
			sep, seed, 
			parallel, 
			null);
	}
	
	/**
	 * Actually called internally
	 * @param caller
	 * @param quantile
	 * @return
	 */
	final protected static double autoEstimateBW(MeanShift caller, double quantile) {
		LogTimer timer = new LogTimer();
		NearestNeighbors nn = new NearestNeighbors(caller, 
				new NearestNeighborsParameters((int)(caller.data.getRowDimension() * quantile))
					.setForceParallel(caller.parallel)).fit();
		caller.info("fit nearest neighbors model for auto-bandwidth automation in " + timer.toString());
		
		return autoEstimateBW(nn,
				caller.data.getDataRef(), quantile, caller.getSeparabilityMetric(), 
					caller.getSeed(), caller.parallel, caller);
	}
	
	final protected static double autoEstimateBW(NearestNeighbors nn, double[][] data, 
			double quantile, GeometricallySeparable sep, Random seed, boolean parallel,
			Loggable logger) {

		if(quantile <= 0 || quantile > 1)
			throw new IllegalArgumentException("illegal quantile");
		final int m = data.length;
		
		double bw = 0.0;
		final double[][] X = nn.data.getDataRef();
		final int minsize = ParallelChunkingTask.ChunkingStrategy.DEF_CHUNK_SIZE;
		final int chunkSize = X.length < minsize ? minsize : X.length / 5;
		final int numChunks = ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, m);
		Neighborhood neighb;
		
		
		if(!parallel) {
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
				for(double[] distRow: neighb.getDistances()) {
					//bw += VecUtils.max(distRow);
					bw += distRow[distRow.length - 1]; // it's sorted!
				}
			}
		} else {
			// Estimate bandwidth in parallel
			bw = ParallelBandwidthEstimator.doAll(X, nn);
		}
		
		return bw / (double)m;
	}
	
	
	/**
	 * Estimates the bandwidth of the model in parallel for scalability
	 * @author Taylor G Smith
	 */
	static class ParallelBandwidthEstimator 
			extends ParallelChunkingTask<Double> 
			implements java.io.Serializable {
		
		private static final long serialVersionUID = 1171269106158790138L;
		final NearestNeighbors nn;
		final int high;
		final int low;
		
		ParallelBandwidthEstimator(double[][] X, NearestNeighbors nn) {
			
			// Use the SimpleChunker
			super(X);
			
			this.nn = nn;
			this.low = 0;
			this.high = strategy.getNumChunks(X);
		}
		
		ParallelBandwidthEstimator(ParallelBandwidthEstimator task, int low, int high) {
			super(task);

			this.nn = task.nn;
			this.low = low;
			this.high = high;
		}

		@Override
		protected Double compute() {
			if(high - low <= 1) { // generally should equal one...
				return reduce(chunks.get(low));
			} else {
				int mid = this.low + (this.high - this.low) / 2;
				ParallelBandwidthEstimator left = new ParallelBandwidthEstimator(this, low, mid);
				ParallelBandwidthEstimator right = new ParallelBandwidthEstimator(this, mid, high);
				
	            left.fork();
	            Double l = right.compute();
	            Double r = left.join();

	            return l + r;
			}
		}

		@Override
		public Double reduce(Chunk chunk) {
			double bw = 0.0;
			Neighborhood neighb = nn.getNeighbors(chunk.get(), false);
			
			for(double[] distRow: neighb.getDistances()) {
				//bw += VecUtils.max(distRow);
				bw += distRow[distRow.length - 1]; // it's sorted!
			}
			
			return bw;
		}
		
		static double doAll(double[][] X, NearestNeighbors nn) {
			return getThreadPool().invoke(new ParallelBandwidthEstimator(X, nn));
		}
	}
	
	
	
	

	/**
	 * Handles the output for the {@link #singleSeed(double[], RadiusNeighbors, double[][], int)}
	 * method. Implements comparable to be sorted by the value in the entry pair.
	 * @author Taylor G Smith
	 */
	protected static class MeanShiftSeed implements Comparable<MeanShiftSeed> {
		final double[] dists;
		/** The number of points in the bandwidth */
		final Integer count;
		final int iterations;
		
		MeanShiftSeed(final double[] dists, final int count, int iterations) {
			this.dists = dists;
			this.count = count;
			this.iterations = iterations;
		}
		
		/*
		 * we don't need these methods in the actual algo, and they just
		 * create more need for testing to get good coverage, so we can
		 * just omit them
		 * 
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
		public String toString() {
			return "{" + Arrays.toString(dists) + " : " + count + "}";
		}
		
		@Override
		public int hashCode() {
			int h = 31;
			for(double d: dists)
				h ^= (int)d;
			return h ^ count;
		}
		*/
		
		EntryPair<double[],Integer> getPair() {
			return new EntryPair<>(dists, count);
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
	

	/**
	 * Light struct to hold summary info
	 * @author Taylor G Smith
	 */
	static class SummaryLite {
		final String name;
		final int iters;
		final String fmtTime;
		final String wallTime;
		boolean retained = false;
		
		SummaryLite(final String nm, final int iter,
				final String fmt, final String wall) {
			this.name = nm;
			this.iters = iter;
			this.fmtTime = fmt;
			this.wallTime = wall;
		}
		
		Object[] toArray() {
			return new Object[]{
				name,
				iters,
				fmtTime,
				wallTime,
				retained
			};
		}
	}
	
	/**
	 * The superclass for parallelized MeanShift tasks
	 * @author Taylor G Smith
	 * @param <T>
	 */
	abstract static class ParallelMSTask<T> extends ParallelChunkingTask<T> {
		private static final long serialVersionUID = 2139716909891672022L;
		final ConcurrentLinkedDeque<SummaryLite> summaries;
		final double[][] X;

		ParallelMSTask(double[][] X, ConcurrentLinkedDeque<SummaryLite> summaries) {
			super(X);
			this.summaries = summaries;
			this.X = X;
		}
		
		ParallelMSTask(ParallelMSTask<T> task) {
			super(task);
			this.summaries = task.summaries;
			this.X = task.X;
		}
		
		public String formatName(String str) {
			StringBuilder sb = new StringBuilder();
			boolean hyphen = false; // have we hit the hyphen yet?
			boolean started_worker = false;
			boolean seen_k = false;
			boolean finished_worker= false;
			
			for(char c: str.toCharArray()) {
				if(hyphen || Character.isUpperCase(c)) {
					if(started_worker && !finished_worker) {
						if(c == 'k') { // past first 'r'...
							seen_k = true;
							continue;
						}
						
						// in the middle of the word "worker"
						if(c != 'r')
							continue;
						else if(!seen_k)
							continue;
						
						// At the last char in 'worker'
						finished_worker = true;
						sb.append("Kernel");
					} else if(!started_worker && c == 'w') {
						started_worker = true;
					} else {
						sb.append(c);
					}
				}
				
				else if('-' == c) {
					hyphen = true;
					sb.append(c);
				}
			}
			
			return sb.toString();
		}
	}
	
	/**
	 * Class that handles construction of the center intensity object
	 * @author Taylor G Smith
	 */
	static abstract class CenterIntensity implements java.io.Serializable, Iterable<MeanShiftSeed> {
		private static final long serialVersionUID = -6535787295158719610L;
		
		abstract int getIters();
		abstract boolean isEmpty();
		abstract ArrayList<SummaryLite> getSummaries();
		abstract int size();
	}
	
	/**
	 * A class that utilizes a {@link java.util.concurrent.ForkJoinPool} 
	 * as parallel executors to run many tasks across multiple cores.
	 * @author Taylor G Smith
	 */
	static class ParallelSeedExecutor 
			extends ParallelMSTask<ConcurrentSkipListSet<MeanShiftSeed>> {
		
		private static final long serialVersionUID = 632871644265502894L;
		
		final int maxIter;
		final RadiusNeighbors nbrs;
		
		final ConcurrentSkipListSet<MeanShiftSeed> computedSeeds;
		final int high, low;
		
		
		ParallelSeedExecutor(
				int maxIter, double[][] X, RadiusNeighbors nbrs,
				ConcurrentLinkedDeque<SummaryLite> summaries) {
			
			/**
			 * Pass summaries reference to super
			 */
			super(X, summaries);
			
			this.maxIter = maxIter;
			this.nbrs = nbrs;
			this.computedSeeds = new ConcurrentSkipListSet<>();
			this.low = 0;
			this.high = strategy.getNumChunks(X);
		}
		
		ParallelSeedExecutor(ParallelSeedExecutor task, int low, int high) {
			super(task);
			
			this.maxIter = task.maxIter;
			this.nbrs = task.nbrs;
			this.computedSeeds = task.computedSeeds;
			this.high = high;
			this.low = low;
		}
		
		@Override
		protected ConcurrentSkipListSet<MeanShiftSeed> compute() {
			if(high - low <= 1) { // generally should equal one...
				return reduce(chunks.get(low));
				
			} else {
				int mid = this.low + (this.high - this.low) / 2;
				ParallelSeedExecutor left  = new ParallelSeedExecutor(this, low, mid);
				ParallelSeedExecutor right  = new ParallelSeedExecutor(this, mid, high);
				
	            left.fork();
	            right.compute();
	            left.join();
	            
	            return computedSeeds;
			}
		}
		
		@Override
		public ConcurrentSkipListSet<MeanShiftSeed> reduce(Chunk chunk) {
			for(double[] seed: chunk.get()) {
				MeanShiftSeed ms = singleSeed(seed, nbrs, X, maxIter);
				if(null == ms)
					continue;
				
				computedSeeds.add(ms);
				String nm = getName();
				summaries.add(new SummaryLite(
					nm, 
					ms.iterations, 
					timer.formatTime(), 
					timer.wallTime()
				));
			}
			
			return computedSeeds;
		}
		
		static ConcurrentSkipListSet<MeanShiftSeed> doAll(
				int maxIter, double[][] X, RadiusNeighbors nbrs,
				ConcurrentLinkedDeque<SummaryLite> summaries) {
			
			return getThreadPool().invoke(
				new ParallelSeedExecutor(
					maxIter, X, nbrs,
					summaries));
		}
	}
	
	class ParallelCenterIntensity extends CenterIntensity {
		private static final long serialVersionUID = 4392163493242956320L;

		final ConcurrentSkipListSet<Integer> itrz = new ConcurrentSkipListSet<>();
		final ConcurrentSkipListSet<MeanShiftSeed> computedSeeds;
		
		/** Serves as a reference for passing to parallel job */
		final ConcurrentLinkedDeque<SummaryLite> summaries = new ConcurrentLinkedDeque<>();
		
		final LogTimer timer;
		final RadiusNeighbors nbrs;
		
		ParallelCenterIntensity(RadiusNeighbors nbrs) {
			
			this.nbrs = nbrs;
			this.timer = new LogTimer();
			
			// Execute forkjoinpool
			this.computedSeeds = ParallelSeedExecutor.doAll(maxIter, seeds, nbrs, summaries);
			for(MeanShiftSeed sd: computedSeeds)
				itrz.add(sd.iterations);
		}

		@Override
		public int getIters() {
			return itrz.last();
		}

		@Override
		public ArrayList<SummaryLite> getSummaries() {
			return new ArrayList<>(summaries);
		}
		
		@Override
		public boolean isEmpty() {
			return computedSeeds.isEmpty();
		}

		@Override
		public Iterator<MeanShiftSeed> iterator() {
			return computedSeeds.iterator();
		}
		
		@Override
		public int size() {
			return computedSeeds.size();
		}
	}
	
	/**
	 * Compute the center intensity entry pairs serially and call the 
	 * {@link MeanShift#singleSeed(double[], RadiusNeighbors, double[][], int)} method
	 * @author Taylor G Smith
	 */
	class SerialCenterIntensity extends CenterIntensity {
		private static final long serialVersionUID = -1117327079708746405L;
		
		int itrz = 0;
		final TreeSet<MeanShiftSeed> computedSeeds;
		final ArrayList<SummaryLite> summaries = new ArrayList<>();
		
		SerialCenterIntensity(RadiusNeighbors nbrs) {
			
			LogTimer timer;
			
			// Now get single seed members
			MeanShiftSeed sd;
			this.computedSeeds = new TreeSet<>();
			final double[][] X = data.getData();
			
			int idx = 0;
			for(double[] seed: seeds) {
				idx++;
				timer = new LogTimer();
				sd = singleSeed(seed, nbrs, X, maxIter);
				
				if(null == sd)
					continue;
				
				computedSeeds.add(sd);
				itrz = FastMath.max(itrz, sd.iterations);
				
				// If it actually converged, add the summary
				summaries.add(new SummaryLite(
					"Kernel "+(idx - 1), sd.iterations, 
					timer.formatTime(), timer.wallTime()
				));
			}
		}

		@Override
		public int getIters() {
			return itrz;
		}

		@Override
		public ArrayList<SummaryLite> getSummaries() {
			return summaries;
		}
		
		@Override
		public boolean isEmpty() {
			return computedSeeds.isEmpty();
		}

		@Override
		public Iterator<MeanShiftSeed> iterator() {
			return computedSeeds.iterator();
		}
		
		@Override
		public int size() {
			return computedSeeds.size();
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
	

	@Override
	protected MeanShift fit() {
		synchronized(fitLock) {
			
			if(null!=labels) // Already fit this model
				return this;
			

			// Put the results into a Map (hash because tree imposes comparable casting)
			final LogTimer timer = new LogTimer();
			centroids = new ArrayList<double[]>();
			
			
			/*
			 * Get the neighborhoods and center intensity object. Will iterate until
			 * either the centers are found, or the max try count is exceeded. For each
			 * iteration, will increase bandwidth.
			 */
			RadiusNeighbors nbrs = new RadiusNeighbors(
				this, bandwidth).fit();
			
			
			// Compute the seeds and center intensity
			// If parallelism is permitted, try it. 
			CenterIntensity intensity = null;
			if(parallel) {
				try {
					intensity = new ParallelCenterIntensity(nbrs);
				} catch(RejectedExecutionException e) {
					// Shouldn't happen...
					warn("parallel search failed; falling back to serial");
				}
			}
			
			// Gets here if serial or if parallel failed...
			if(null == intensity)
				intensity = new SerialCenterIntensity(nbrs);
			
			
			// Check for points all too far from seeds
			if(intensity.isEmpty()) {
				error(new IllegalClusterStateException("No point "
					+ "was within bandwidth="+bandwidth
					+" of any seed; try increasing bandwidth"));
			} else {
				converged = true;
				itersElapsed = intensity.getIters(); // max iters elapsed
			}
			
			
			
			
			// Extract the centroids
			int idx = 0, m_prime = intensity.size();
			final Array2DRowRealMatrix sorted_centers = new Array2DRowRealMatrix(m_prime,n);
			
			for(MeanShiftSeed entry: intensity)
				sorted_centers.setRow(idx++, entry.getPair().getKey());
			
			// Fit the new neighbors model
			nbrs = new RadiusNeighbors(sorted_centers,
				new RadiusNeighborsParameters(bandwidth)
					.setSeed(this.random_state)
					.setMetric(this.dist_metric)
					.setForceParallel(parallel), true).fit();
			
			

			
			// Post-processing. Remove near duplicate seeds
			// If dist btwn two kernels is less than bandwidth, remove one w fewer pts
			// Create a boolean mask, init true
			final boolean[] unique = new boolean[m_prime];
			for(int i = 0; i < unique.length; i++) unique[i] = true;

			
			// Pre-filtered summaries...
			ArrayList<SummaryLite> allSummary = intensity.getSummaries();
			
			
			// Iterate over sorted centers and query radii
			int redundant_ct = 0;
			int[] indcs;
			double[] center;
			for(int i = 0; i < m_prime; i++) {
				if(unique[i]) {
					center = sorted_centers.getRow(i);
					indcs = nbrs.getNeighbors(
						new double[][]{center}, 
						bandwidth, false)
							.getIndices()[0];
					
					for(int id: indcs)
						unique[id] = false;
					
					unique[i] = true; // Keep this as true
				}
			}
			
			
			// Now assign the centroids...
			SummaryLite summ;
			for(int i = 0; i < unique.length; i++) {
				summ = allSummary.get(i);
				
				if(unique[i]) {
					summ.retained = true;
					centroids.add(sorted_centers.getRow(i));
				}
				
				fitSummary.add(summ.toArray());
			}
			
			
			// calc redundant ct
			redundant_ct = unique.length - centroids.size();
			
			
			// also put the centroids into a matrix. We have to
			// wait to perform this op, because we have to know
			// the size of centroids first...
			Array2DRowRealMatrix centers = new Array2DRowRealMatrix(centroids.size(),n);
			for(int i = 0; i < centroids.size(); i++)
				centers.setRow(i, centroids.get(i));
			
			
			// Build yet another neighbors model...
			NearestNeighbors nn = new NearestNeighbors(centers,
				new NearestNeighborsParameters(1)
					.setSeed(this.random_state)
					.setMetric(this.dist_metric)
					.setForceParallel(false), true).fit();
			
			
			
			info((numClusters=centroids.size())+" optimal kernel"+(numClusters!=1?"s":"")+" identified");
			info(redundant_ct+" nearly-identical kernel"+(redundant_ct!=1?"s":"") + " removed");
			
			
			// Get the nearest...
			final LogTimer clustTimer = new LogTimer();
			Neighborhood knrst = nn.getNeighbors(data.getDataRef());
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
		}
		
	} // End train


	@Override
	public ArrayList<double[]> getCentroids() {
		if(null != centroids) {
			final ArrayList<double[]> cent = new ArrayList<double[]>();
			for(double[] d : centroids)
				cent.add(VecUtils.copy(d));
			
			return cent;
		} else {
			error(new ModelNotFitException("model has not yet been fit"));
			return null; // can't happen
		}
	}

	@Override
	public int[] getLabels() {
		return super.handleLabelCopy(labels);
	}
	
	static MeanShiftSeed singleSeed(double[] seed, RadiusNeighbors rn, double[][] X, int maxIter) {
		final double bandwidth = rn.getRadius(), tolerance = 1e-3;
		final int n = X[0].length; // we know X is uniform
		int completed_iterations = 0;
		
		double norm, diff;
		
		while(true) {

			Neighborhood nbrs = rn.getNeighbors(new double[][]{seed}, bandwidth, false);
			int[] i_nbrs = nbrs.getIndices()[0];
			
			// Check if exit
			if(i_nbrs.length == 0) 
				break;
			
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
			if( completed_iterations++ == maxIter || norm < tolerance )
				return new MeanShiftSeed(seed, i_nbrs.length, completed_iterations);
		}
		
		// Default... shouldn't get here though
		return null;
	}
	
	

	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Seed ID","Iterations","Iter. Time","Wall","Retained"
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
	
	/** {@inheritDoc} */
	@Override
	public int[] predict(RealMatrix newData) {
		return CentroidUtils.predict(this, newData);
	}
}
