package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.scoring.SilhouetteScore;
import com.clust4j.metrics.scoring.UnsupervisedIndexAffinity;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.Named;
import com.clust4j.utils.VecUtils;

public abstract class AbstractCentroidClusterer extends AbstractPartitionalClusterer 
		implements CentroidLearner, Convergeable, UnsupervisedClassifier {
	
	private static final long serialVersionUID = -424476075361612324L;
	final public static double DEF_CONVERGENCE_TOLERANCE = 0.005; // Not same as Convergeable.DEF_TOL
	final public static int DEF_K = Neighbors.DEF_K;
	final public static InitializationStrategy DEF_INIT = InitializationStrategy.KM_AUGMENTED;
	
	final protected InitializationStrategy init;
	final protected int maxIter;
	final protected double tolerance;
	final protected int[] init_centroid_indices;
	final protected int m;
	
	volatile protected boolean converged = false;
	volatile protected double cost = Double.NaN;
	volatile protected int[] labels = null;
	volatile protected int iter = 0;
	
	/** Key is the group label, value is the corresponding centroid */
	volatile protected ArrayList<double[]> centroids = new ArrayList<double[]>();
	volatile protected TreeMap<Integer, ArrayList<Integer>> cent_to_record = null;

	
	static interface Initializer { int[] getInitialCentroidSeeds(double[][] X, int k, final Random seed); }
	public static enum InitializationStrategy implements java.io.Serializable, Initializer, Named {
		/**
		 * Initialize {@link KMeans} or {@link KMedoids} with a set of randomly
		 * selected centroids to use as the initial seeds. This is the traditional
		 * initialization procedure in both KMeans and KMedoids and typically performs
		 * worse than using {@link InitializationStrategy#KM_AUGMENTED}
		 */
		RANDOM {
			@Override public int[] getInitialCentroidSeeds(double[][] X, int k, final Random seed) {
				final int m = X.length;
				
				// Corner case: k = m
				if(m == k)
					return VecUtils.arange(k);
				
				final int[] recordIndices = VecUtils.permutation(VecUtils.arange(m), seed);
				final int[] cent_indices = new int[k];
				for(int i = 0; i < k; i++)
					cent_indices[i] = recordIndices[i];
				return cent_indices;
			}
			
			@Override public String getName() {
				return "random initialization";
			}
		},
		
		/**
		 * Proposed in 2007 by David Arthur and Sergei Vassilvitskii, this <i>k</i>-means++ initialization
		 * algorithms is an approximation algorithm for the NP-hard k-means problem - a way of avoiding the 
		 * sometimes poor clusterings found by the standard k-means algorithm.
		 * @see <a href="https://en.wikipedia.org/wiki/K-means%2B%2B">k-means++</a>
		 * @see <a href="http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf">k-means++ paper</a>
		 */
		KM_AUGMENTED {
			@Override public int[] getInitialCentroidSeeds(double[][] X, int k, final Random seed) {
				final int m = X.length, n = X[0].length;
				final int[] range = VecUtils.arange(k);
				final double[][] centers = new double[k][n];
				final int[] centerIdcs = new int[k];
				
				
				// Corner case: k = m
				if(m == k)
					return range;
				
				// First need to get row norms, which is equal to X * X => row sums
				// True Euclidean norm would sqrt each term, but no need...
				final double[] norms = new double[m];
				for(int i = 0; i < m; i++)
					for(int j = 0; j < X[i].length; j++)
						norms[i] += X[i][j] * X[i][j];
				
				// Arthur and Vassilvitskii reported that this helped
				final int numTrials = FastMath.max(2 * (int)FastMath.log(k), 1);
				
				
				// Start with a random center
				int center_id = seed.nextInt(m);
				centers[0] = X[center_id];
				centerIdcs[0] = center_id;
				
				// Initialize list of closest distances
				double[][] closest = eucDists(new double[][]{centers[0]}, X);
				double currentPotential = MatUtils.sum(closest);
				
				
				// Pick the rest of the cluster starting points
				double[] randomVals, cumSum;
				int[] candidateIdcs;
				double[][] candidateRows, distsToCandidates, bestDistSq;
				int bestCandidate;
				double bestPotential;
				
				
				for(int i = 1; i < k; i++) { // if k == 1, will skip this
					
					/* 
					 * Generate some random vals. This is a precursor to choosing
					 * centroid candidates by sampling with probability proportional to
					 * partial distance to nearest existing centroid
					 */
					randomVals = new double[numTrials];
					for(int j = 0; j < randomVals.length; j++)
						randomVals[j] = currentPotential * seed.nextDouble();
					
					
					/* Search sorted and get new dists for candidates */
					cumSum = MatUtils.cumSum(closest); // always will be sorted
					candidateIdcs = searchSortedCumSum(cumSum, randomVals);
					
					// Identify the candidates
					candidateRows = new double[candidateIdcs.length][];
					for(int j = 0; j < candidateRows.length; j++)
						candidateRows[j] = X[candidateIdcs[j]];
					
					// dists to candidates
					distsToCandidates = eucDists(candidateRows, X);
					
					
					// Identify best candidate...
					bestCandidate	= -1;
					bestPotential	= Double.POSITIVE_INFINITY;
					bestDistSq		= null;
					
					for(int trial = 0; trial < numTrials; trial++) {
						double[] trialCandidate = distsToCandidates[trial];
						double[][] newDistSq = new double[closest.length][trialCandidate.length];
						
						// Build min dist array
						double newPotential = 0.0; // running sum
						for(int j = 0; j < newDistSq.length; j++) {
							for(int p = 0; p < trialCandidate.length; p++) {
								newDistSq[j][p] = FastMath.min(closest[j][p], trialCandidate[p]);
								newPotential += newDistSq[j][p];
							}
						}
						
						// Store if best so far
						if(-1 == bestCandidate || newPotential < bestPotential) {
							bestCandidate = candidateIdcs[trial];
							bestPotential = newPotential;
							bestDistSq = newDistSq;
						}
					}
					
					
					// Add the record...
					centers[i] 		= X[bestCandidate];
					centerIdcs[i] 	= bestCandidate;
					
					// update vars outside loop
					currentPotential = bestPotential;
					closest = bestDistSq;
				}
				
				
				return centerIdcs;
			}
			
			@Override public String getName() {
				return "k-means++";
			}
		}
	}
	
	/** Internal method for cumsum searchsorted. Protected for testing only */
	static int[] searchSortedCumSum(double[] cumSum, double[] randomVals) {
		final int[] populate = new int[randomVals.length];
		
		for(int c = 0; c < populate.length; c++) {
			populate[c] = cumSum.length - 1;
			
			for(int cmsm = 0; cmsm < cumSum.length; cmsm++) {
				if(randomVals[c] <= cumSum[cmsm]) {
					populate[c] = cmsm;
					break;
				}
			}
		}
		
		return populate;
	}
	
	/** Internal method for computing candidate distances. Protected for testing only */
	static double[][] eucDists(double[][] centers, double[][] X) {
		MatUtils.checkDimsForUniformity(X);
		MatUtils.checkDimsForUniformity(centers);
		
		final int m = X.length, n = X[0].length;
		if(n != centers[0].length)
			throw new DimensionMismatchException(n, centers[0].length);
		
		int next = 0;
		final double[][] dists = new double[centers.length][m];
		for(double[] d: centers) {
			for(int i = 0; i < m; i++)
				dists[next][i] = Distance.EUCLIDEAN.getPartialDistance(d, X[i]);
			next++;
		}
		
		return dists;
	}
	
	
	
	public AbstractCentroidClusterer(AbstractRealMatrix data,
			CentroidClustererPlanner planner) {
		super(data, planner, planner.getK());
		
		this.init = planner.getInitializationStrategy();
		this.maxIter = planner.getMaxIter();
		this.tolerance = planner.getConvergenceTolerance();
		this.m = data.getRowDimension();
		
		if(maxIter < 0)	throw new IllegalArgumentException("maxIter must exceed 0");
		if(tolerance<0)	throw new IllegalArgumentException("minChange must exceed 0");

		
		// set centroids
		final LogTimer centTimer = new LogTimer();
		this.init_centroid_indices = init.getInitialCentroidSeeds(
			this.data.getData(), k, getSeed());
		for(int i: this.init_centroid_indices)
			centroids.add(this.data.getRow(i));
		
		
		info("selected centroid centers via " + init.getName() + " in " + centTimer.toString());
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","K","Scale","Force Par.","Allow Par.","Max Iter","Tolerance","Init."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),k,normalized,
				GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE,
				GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM,
				maxIter, tolerance, init.toString()
			});
	}

	
	
	public static abstract class CentroidClustererPlanner 
			extends BaseClustererPlanner 
			implements UnsupervisedClassifierPlanner, ConvergeablePlanner {
		private static final long serialVersionUID = -1984508955251863189L;
		
		abstract public int getK();
		@Override abstract public int getMaxIter();
		@Override abstract public double getConvergenceTolerance();
		abstract public InitializationStrategy getInitializationStrategy();
		abstract public CentroidClustererPlanner setConvergenceCriteria(final double min);
		abstract public CentroidClustererPlanner setInitializationStrategy(final InitializationStrategy strat);
	}
	



	/**
	 * Returns a matrix with a reference to centroids. Use with care.
	 * @return Array2DRowRealMatrix
	 */
	protected Array2DRowRealMatrix centroidsToMatrix() {
		double[][] c = new double[k][];
		
		int i = 0;
		for(double[] row: centroids)
			c[i++] = row;
		
		return new Array2DRowRealMatrix(c, false);
	}
	
	@Override
	public boolean didConverge() {
		return converged;
	}
	
	@Override
	public ArrayList<double[]> getCentroids() {
		final ArrayList<double[]> cent = new ArrayList<double[]>();
		for(double[] d : centroids)
			cent.add(VecUtils.copy(d));
		
		return cent;
	}
	
	/**
	 * Returns a copy of the classified labels
	 */
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
	public int getMaxIter() {
		return maxIter;
	}
	
	@Override
	public double getConvergenceTolerance() {
		return tolerance;
	}
	
	/**
	 * In the corner case that k = 1, the {@link LabelEncoder}
	 * won't work, so we need to label everything as 0 and immediately return
	 */
	final void labelFromSingularK(final double[][] X) {
		labels = VecUtils.repInt(0, m);
		double[] center_record = MatUtils.meanRecord(X);
		
		cost = 0;
		double diff;
		for(double[] d: X) {
			for(int j = 0; j < data.getColumnDimension(); j++) {
				diff = d[j] - center_record[j];
				cost += diff * diff;
			}
		}
		
		iter++;
		converged = true;
		warn("k=1; converged immediately with a TSS of "+cost);
	}
	
	@Override
	public int itersElapsed() {
		return iter;
	}
	
	/** {@inheritDoc} */
	@Override
	public double indexAffinityScore(int[] labels) {
		// Propagates ModelNotFitException
		return UnsupervisedIndexAffinity.getInstance().evaluate(labels, getLabels());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		return silhouetteScore(getSeparabilityMetric());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore(GeometricallySeparable dist) {
		// Propagates ModelNotFitException
		return SilhouetteScore.getInstance().evaluate(this, dist, getLabels());
	}
	
	/**
	 * Return the cost of the entire clustering system. For KMeans, this
	 * equates to total sum of squares
	 * @return system cost
	 */
	public double totalCost() {
		return cost;
	}
	
	/**
	 * Reorder the labels in order of appearance using the 
	 * {@link LabelEncoder}. Also reorder the centroids to correspond
	 * with new label order
	 */
	void reorderLabelsAndCentroids() {
		if(null == labels)
			throw new ModelNotFitException("model not yet fit");
		
		final LabelEncoder encoder = new LabelEncoder(labels).fit();
		labels =  encoder.getEncodedLabels();
		
		// also reorder centroids... takes O(2K) passes
		TreeMap<Integer, double[]> tmpCentroids = new TreeMap<>(); 
		// tm seems like overkill, but since KMedoids labels using the
		// medoid index, we'll either get NPEs or AIOOBEs if we use the array:
		// final double[][] tmpCentroids = new double[k][];
		
		int j = 0;
		for(int i: encoder.getClasses())
			tmpCentroids.put(encoder.encodeOrNull(i), centroids.get(j++));
		
		for(int i = 0; i < k; i++)
			centroids.set(i, tmpCentroids.get(i));
	}
	
	/**
	 * For computing the within sum of squares
	 * @param instances
	 * @param centroid
	 * @return
	 */
	static double barycentricDistance(double[][] instances, double[] centroid) {
		double clust_cost = 0.0, diff;
		final int n = centroid.length;
		
		for(double[] instance: instances) {
			if(n != instance.length)
				throw new DimensionMismatchException(n, instance.length);
			
			for(int j = 0; j < n; j++) {
				diff = instance[j] - centroid[j];
				clust_cost += diff * diff;
			}
		}
		
		return clust_cost;
	}
}
