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
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.NamedEntity;
import com.clust4j.kernel.Kernel;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

import static com.clust4j.metrics.scoring.UnsupervisedMetric.SILHOUETTE;

public abstract class AbstractCentroidClusterer extends AbstractPartitionalClusterer 
		implements CentroidLearner, Convergeable, UnsupervisedClassifier {
	
	private static final long serialVersionUID = -424476075361612324L;
	final public static double DEF_CONVERGENCE_TOLERANCE = 0.005; // Not same as Convergeable.DEF_TOL
	final public static int DEF_K = BaseNeighborsModel.DEF_K;
	final public static InitializationStrategy DEF_INIT = InitializationStrategy.AUTO;
	final public static HashSet<Class<? extends GeometricallySeparable>> UNSUPPORTED_METRICS;
	
	static {
		UNSUPPORTED_METRICS = new HashSet<>();
		
		/*
		 * Add all binary distances
		 */
		for(Distance d: Distance.binaryDistances())
			UNSUPPORTED_METRICS.add(d.getClass());
		
		/*
		 * Kernels that conditional positive def or 
		 * may propagate NaNs or Infs or 100% zeros
		 */
		
		// should be handled now by returning just one cluster...
		//UNSUPPORTED_METRICS.add(CauchyKernel.class);
		//UNSUPPORTED_METRICS.add(CircularKernel.class);
		//UNSUPPORTED_METRICS.add(GeneralizedMinKernel.class);
		//UNSUPPORTED_METRICS.add(HyperbolicTangentKernel.class);
		//UNSUPPORTED_METRICS.add(InverseMultiquadricKernel.class);
		//UNSUPPORTED_METRICS.add(LogKernel.class);
		//UNSUPPORTED_METRICS.add(MinKernel.class);
		//UNSUPPORTED_METRICS.add(MultiquadricKernel.class);
		//UNSUPPORTED_METRICS.add(PolynomialKernel.class);
		//UNSUPPORTED_METRICS.add(PowerKernel.class);
		//UNSUPPORTED_METRICS.add(SplineKernel.class);
	}
	
	
	protected InitializationStrategy init;
	final protected int maxIter;
	final protected double tolerance;
	final protected int[] init_centroid_indices;
	final protected int m;
	
	volatile protected boolean converged = false;
	volatile protected double tss = 0.0;
	volatile protected double bss = Double.NaN;
	volatile protected double[] wss;
	
	volatile protected int[] labels = null;
	volatile protected int iter = 0;
	
	/** Key is the group label, value is the corresponding centroid */
	volatile protected ArrayList<double[]> centroids = new ArrayList<double[]>();

	
	static interface Initializer { int[] getInitialCentroidSeeds(AbstractCentroidClusterer model, double[][] X, int k, final Random seed); }
	public static enum InitializationStrategy implements java.io.Serializable, Initializer, NamedEntity {
		AUTO {
			@Override public int[] getInitialCentroidSeeds(AbstractCentroidClusterer model, double[][] X, int k, final Random seed) {
				if(model.dist_metric instanceof Kernel)
					return RANDOM.getInitialCentroidSeeds(model, X, k, seed);
				return KM_AUGMENTED.getInitialCentroidSeeds(model, X, k, seed);
			}
			
			@Override public String getName() {
				return "auto initialization";
			}
		},
		
		/**
		 * Initialize {@link KMeans} or {@link KMedoids} with a set of randomly
		 * selected centroids to use as the initial seeds. This is the traditional
		 * initialization procedure in both KMeans and KMedoids and typically performs
		 * worse than using {@link InitializationStrategy#KM_AUGMENTED}
		 */
		RANDOM {
			@Override public int[] getInitialCentroidSeeds(AbstractCentroidClusterer model, double[][] X, int k, final Random seed) {
				model.init = this;
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
			@Override public int[] getInitialCentroidSeeds(AbstractCentroidClusterer model, double[][] X, int k, final Random seed) {
				model.init = this;
				
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
	
	
	
	public AbstractCentroidClusterer(RealMatrix data,
			CentroidClustererParameters<? extends AbstractCentroidClusterer> planner) {
		super(data, planner, planner.getK());
		
		/*
		 * Check for prohibited dist metrics...
		 */
		if( !isValidMetric(this.dist_metric) ) {
			warn(this.dist_metric.getName() + " is unsupported by "+getName()+"; "
					+ "falling back to default (" + defMetric().getName() + ")");
			
			/*
			 * If this is KMedoids, we set it to Mahattan, otherwise Euclidean
			 */
			this.setSeparabilityMetric(defMetric());
		}
		
		this.init = planner.getInitializationStrategy();
		this.maxIter = planner.getMaxIter();
		this.tolerance = planner.getConvergenceTolerance();
		this.m = data.getRowDimension();
		
		if(maxIter < 0)	throw new IllegalArgumentException("maxIter must exceed 0");
		if(tolerance<0)	throw new IllegalArgumentException("minChange must exceed 0");

		
		// set centroids
		final LogTimer centTimer = new LogTimer();
		this.init_centroid_indices = init.getInitialCentroidSeeds(
			this, this.data.getData(), k, getSeed());
		for(int i: this.init_centroid_indices)
			centroids.add(this.data.getRow(i));
		
		
		info("selected centroid centers via " + init.getName() + " in " + centTimer.toString());
		logModelSummary();
		
		/*
		 * The TSS will always be the same -- the sum of squared distances from the mean record.
		 * We can just compute this here quick and easy.
		 */
		final double[][] X = this.data.getDataRef();
		final double[] mean_record = MatUtils.meanRecord(X);
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < mean_record.length; j++){
				double diff = X[i][j] - mean_record[j];
				tss += (diff * diff);
			}
		}
		
		// Initialize WSS:
		wss = VecUtils.rep(Double.NaN, k);
	}
	
	@Override
	final public boolean isValidMetric(GeometricallySeparable geo) {
		return !UNSUPPORTED_METRICS.contains(geo.getClass());
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","K","Allow Par.","Max Iter","Tolerance","Init."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),k,
				parallel,
				maxIter, tolerance, init.toString()
			});
	}

	
	

	
	@Override
	public boolean didConverge() {
		synchronized(fitLock) {
			return converged;
		}
	}
	
	@Override
	public ArrayList<double[]> getCentroids() {
		synchronized(fitLock) {
			final ArrayList<double[]> cent = new ArrayList<double[]>();
			for(double[] d : centroids)
				cent.add(VecUtils.copy(d));
			
			return cent;
		}
	}
	
	/**
	 * Returns a copy of the classified labels
	 */
	@Override
	public int[] getLabels() {
		synchronized(fitLock) {
			return super.handleLabelCopy(labels);
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
	protected final void labelFromSingularK(final double[][] X) {
		labels = VecUtils.repInt(0, m);
		wss = new double[]{tss};
		iter++;
		converged = true;
		warn("k=1; converged immediately with a TSS of "+tss);
	}
	
	@Override
	public int itersElapsed() {
		synchronized(fitLock) {
			return iter;
		}
	}
	
	/** {@inheritDoc} */
	@Override
	public double indexAffinityScore(int[] labels) {
		// Propagates ModelNotFitException
		return SupervisedMetric.INDEX_AFFINITY.evaluate(labels, getLabels());
	}
	
	/** {@inheritDoc} */
	@Override
	public int[] predict(RealMatrix newData) {
		return CentroidUtils.predict(this, newData);
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		// Propagates ModelNotFitException
		return SILHOUETTE.evaluate(this, getLabels());
	}
	
	
	public double getTSS() {
		// doesn't need to be synchronized, because
		// calculated in the constructor always
		return tss;
	}
	
	public double[] getWSS() {
		synchronized(fitLock) {
			if(null == wss) {
				return VecUtils.rep(Double.NaN, k);
			} else {
				return VecUtils.copy(wss);
			}
		}
	}
	
	public double getBSS() {
		synchronized(fitLock) {
			return bss;
		}
	}

	protected abstract void reorderLabelsAndCentroids();
	@Override protected abstract AbstractCentroidClusterer fit();
	protected GeometricallySeparable defMetric() { return AbstractClusterer.DEF_DIST; }
}
