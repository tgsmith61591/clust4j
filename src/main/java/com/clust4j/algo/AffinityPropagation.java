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

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.Axis;

/**
 * <a href="https://en.wikipedia.org/wiki/Affinity_propagation">Affinity Propagation</a> (AP) 
 * is a clustering algorithm based on the concept of "message passing" between data points. 
 * Unlike other clustering algorithms such as {@link KMeans} or {@link KMedoids}, 
 * AP does not require the number of clusters to be determined or estimated before 
 * running the algorithm. Like KMedoids, AP finds "exemplars", members of the input 
 * set that are representative of clusters.
 * 
 * @see <a href="https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/affinity_propagation_.py">sklearn</a>
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;, adapted from sklearn Python implementation
 *
 */
final public class AffinityPropagation extends AbstractAutonomousClusterer implements Convergeable, CentroidLearner {
	private static final long serialVersionUID = 1986169131867013043L;
	
	/** The number of stagnant iterations after which the algorithm will declare convergence */
	final public static int DEF_ITER_BREAK = 15;
	final public static int DEF_MAX_ITER = 200;
	final public static double DEF_DAMPING = 0.5;
	/** By default uses minute Gaussian smoothing. It is recommended this remain
	 *  true, but the {@link AffinityPropagationParameters#useGaussianSmoothing(boolean)}
	 *  method can disable this option */
	final public static boolean DEF_ADD_GAUSSIAN_NOISE = true;
	final public static HashSet<Class<? extends GeometricallySeparable>> UNSUPPORTED_METRICS;
	
	
	/**
	 * Static initializer
	 */
	static {
		UNSUPPORTED_METRICS = new HashSet<>();
		
		/*
		 *  can produce negative inf, but should be OK:
		 *  UNSUPPORTED_METRICS.add(CircularKernel.class); 
		 *  UNSUPPORTED_METRICS.add(LogKernel.class);
		 */
		
		// Add more metrics here if necessary...
	}
	
	@Override final public boolean isValidMetric(GeometricallySeparable geo) {
		return !UNSUPPORTED_METRICS.contains(geo.getClass());
	}
	
	
	
	/** Damping factor */
	private final double damping;
	
	/** Remove degeneracies with noise? */
	private final boolean addNoise;
	
	/** Number of stagnant iters after which to break */
	private final int iterBreak;
	
	/** The max iterations */
	private final int maxIter;

	/** Num rows, cols */
	private final int m;
	
	/** Min change convergence criteria */
	private final double tolerance;
	
	/** Class labels */
	private volatile int[] labels = null;
	
	/** Track convergence */
	private volatile boolean converged = false;
	
	/** Number of identified clusters */
	private volatile int numClusters;
	
	/** Count iterations */
	private volatile int iterCt = 0;
	
	/** Sim matrix. Only use during fitting, then back to null to save space */
	private volatile double[][] sim_mat = null;
	
	/** Holds the centroids */
	private volatile ArrayList<double[]> centroids = null;
	
	/** Holds centroid indices */
	private volatile ArrayList<Integer> centroidIndices = null;
	
	/** Holds the availability matrix */
	volatile private double[][] cachedA;
	
	/** Holds the responsibility matrix */
	volatile private double[][] cachedR;
	
	
	
	
	/**
	 * Initializes a new AffinityPropagationModel with default parameters
	 * @param data
	 */
	protected AffinityPropagation(final RealMatrix data) {
		this(data, new AffinityPropagationParameters());
	}
	
	/**
	 * Initializes a new AffinityPropagationModel with parameters
	 * @param data
	 * @param planner
	 */
	public AffinityPropagation(final RealMatrix data, final AffinityPropagationParameters planner) {
		super(data, planner);
		
		
		// Check some args
		if(planner.damping < DEF_DAMPING || planner.damping >= 1)
			error(new IllegalArgumentException("damping "
				+ "must be between " + DEF_DAMPING + " and 1"));
		
		this.damping = planner.damping;
		this.iterBreak = planner.iterBreak;
		this.m = data.getRowDimension();
		this.tolerance = planner.minChange;
		this.maxIter = planner.maxIter;
		this.addNoise = planner.addNoise;
		
		if(maxIter < 0)	throw new IllegalArgumentException("maxIter must exceed 0");
		if(tolerance<0)	throw new IllegalArgumentException("minChange must exceed 0");
		if(iterBreak<0)	throw new IllegalArgumentException("iterBreak must exceed 0");
		
		if(!addNoise) {
			warn("not scaling with Gaussian noise can cause the algorithm not to converge");
		}
		
		/*
		 * Shouldn't be an issue with AP
		 */
		if(!isValidMetric(this.dist_metric)) {
			warn(this.dist_metric.getName() + " is not valid for "+getName()+". "
				+ "Falling back to default Euclidean dist");
			setSeparabilityMetric(DEF_DIST);
		}
		
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Damping","Allow Par.","Max Iter","Tolerance","Add Noise"
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),damping,
				parallel,
				maxIter, tolerance, addNoise
			});
	}



	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof AffinityPropagation) {
			AffinityPropagation a = (AffinityPropagation)o;
			
			/*
			 * This should apply to cachedR as well, so no
			 * need to check that lest we uselessly impose
			 * less coverage. This is also a litmus test of
			 * whether the model has been fit yet.
			 */
			if(null == this.cachedA ^ null == a.cachedA)
				return false;
			
			return super.equals(o) // check on UUID and class
				&& MatUtils.equalsExactly(this.data.getDataRef(), a.data.getDataRef())
				&& VecUtils.equalsExactly(this.labels, a.labels)
				&& this.tolerance == a.tolerance
				&& this.addNoise == a.addNoise
				&& this.maxIter == a.maxIter
				&& this.damping == a.damping;
		}
		
		return false;
	}

	@Override
	public int[] getLabels() {
		return super.handleLabelCopy(labels);
	}

	@Override
	public boolean didConverge() {
		return converged;
	}
	
	public double[][] getAvailabilityMatrix() {
		if(null != cachedA)
			return MatUtils.copy(cachedA);
		throw new ModelNotFitException("model is not fit");
	}
	
	public double[][] getResponsibilityMatrix() {
		if(null != cachedR)
			return MatUtils.copy(cachedR);
		throw new ModelNotFitException("model is not fit");
	}

	@Override
	public int getMaxIter() {
		return maxIter;
	}

	@Override
	public double getConvergenceTolerance() {
		return tolerance;
	}

	@Override
	public int itersElapsed() {
		return iterCt;
	}

	@Override
	public String getName() {
		return "AffinityPropagation";
	}

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AFFINITY_PROP;
	}
	
	/**
	 * Remove this from scope of {@link #fit()} to avoid lots of large objects
	 * left in memory. This is more space efficient and promotes easier testing.
	 * @param X
	 * @param metric
	 * @param seed
	 * @param addNoise
	 * @return the smoothed similarity matrix
	 */
	protected static double[][] computeSmoothedSimilarity(final double[][] X, GeometricallySeparable metric, Random seed, boolean addNoise) {
		/*
		 * Originally, we computed similarity matrix, then refactored the diagonal vector, and
		 * then computed the following portions. We can do this all at once and save lots of passes
		 * (5?) on the order of O(M^2), condensing it all to one pass of O(M choose 2).
		 * 
		 * After the sim matrix is computed, we need to do three things:
		 * 
		 * 1. Create a matrix of very small values (tiny_scaled) to remove degeneracies in sim_mal
		 * 2. Multiply tiny_scaled by an extremely small value (GlobalState.Mathematics.TINY*100)
		 * 3. Create a noise matrix of random Gaussian values and add it to the similarity matrix.
		 * 
		 * The methods exist to build these in three to five separate O(M^2) passes, but that's 
		 * extremely expensive, so we're going to do it in one giant, convoluted loop. If you're 
		 * trying to debug this, sorry...
		 * 
		 * Total runtime: O(2M * M choose 2)
		 */
		final int m = X.length;
		double[][] sim_mat = new double[m][m];
		
		int idx = 0;
		final double tiny_val = GlobalState.Mathematics.TINY*100;
		final double[] vector = new double[m * m];
		double sim, noise;
		boolean last_iter = false;
		
		
		// Do this a little differently... set the diagonal FIRST.
		for(int i = 0; i < m; i++) {
			sim = -(metric.getPartialDistance(X[i], X[i]));
			sim_mat[i][i] = sim;
			vector[idx++] = sim;
		}
		
		
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) { // Upper triangular
				sim = -(metric.getPartialDistance(X[i], X[j])); // similarity
				
				// Assign to upper and lower portion
				sim_mat[i][j] = sim;
				sim_mat[j][i] = sim;
				
				// Add to the vector (twice)
				for(int b = 0; b < 2; b++)
					vector[idx++] = sim;
				
				// Catch the last iteration, compute the pref:
				double median = 0.0;
				if(last_iter = (i == m - 2 && j == m - 1))
					median = VecUtils.median(vector);
				
				if(addNoise) {
					noise = (sim * GlobalState.Mathematics.EPS + tiny_val);
					sim_mat[i][j] += (noise * seed.nextGaussian());
					sim_mat[j][i] += (noise * seed.nextGaussian());
					
					if(last_iter) { // set diag and do the noise thing.
						noise = (median * GlobalState.Mathematics.EPS + tiny_val);
						for(int h = 0; h < m; h++)
							sim_mat[h][h] = median + (noise * seed.nextGaussian());
					}
				} else if(last_iter) {
					// it's the last iter and no noise. Just set diag.
					for(int h = 0; h < m; h++)
						sim_mat[h][h] = median;
				}
			}
		}
		
		return sim_mat;
	}

	
	/**
	 * Computes the first portion of the AffinityPropagation iteration
	 * sequence in place. Separating this piece from the {@link #fit()} method
	 * itself allows for easier testing.
	 * @param A
	 * @param S
	 * @param tmp
	 * @param I
	 * @param Y
	 * @param Y2
	 */
	protected static void affinityPiece1(double[][] A, double[][] S, double[][] tmp, int[] I, double[] Y, double[] Y2) {
		final int m = S.length;
		
		// Reassign tmp, create vector of arg maxes. Can
		// assign tmp like this:
		//
		//		tmp = MatUtils.add(A, sim_mat);
		//
		//
		// But requires extra M x M pass. Also get indices of ROW max. 
		// Can do like this:
		//
		//		I = MatUtils.argMax(tmp, Axis.ROW);
		//
		// But requires extra pass on order of M. Finally, capture the second
		// highest record in each row, and store in a vector. Then row-wise
		// scalar subtract Y from the sim_mat
		for(int i = 0; i < m; i++) {
			
			// Compute row maxes
			double runningMax = Double.NEGATIVE_INFINITY;
			double secondMax  = Double.NEGATIVE_INFINITY;
			int runningMaxIdx = 0; //-1; // Idx of max row element -- start at 0 in case metric produces -Infs
			
			for(int j = 0; j < m; j++) { 	// Create tmp as A + sim_mat
				tmp[i][j] = A[i][j] + S[i][j];
				
				if(tmp[i][j] > runningMax) {
					secondMax = runningMax;
					runningMax = tmp[i][j];
					runningMaxIdx = j;
				} else if(tmp[i][j] > secondMax) {
					secondMax = tmp[i][j];
				}
			}
			
			I[i] = runningMaxIdx;			// Idx of max element for row
			Y[i] = tmp[i][I[i]]; // Grab the current val
			Y2[i] = secondMax;
			tmp[i][I[i]] = Double.NEGATIVE_INFINITY; // Set that idx to neg inf now
		}
	}
	
	/**
	 * Computes the second portion of the AffinityPropagation iteration
	 * sequence in place. Separating this piece from the {@link #fit()} method
	 * itself allows for easier testing.
	 * @param colSums
	 * @param tmp
	 * @param I
	 * @param S
	 * @param R
	 * @param Y
	 * @param Y2
	 * @param damping
	 */
	protected static void affinityPiece2(double[] colSums, double[][] tmp, int[] I, 
			double[][] S, double[][] R, double[] Y, double[] Y2, double damping) {
		
		final int m = S.length;
		
		// Second i thru m loop, get new max vector and then first damping.
		// First damping ====================================
		// This can be done like this (which is more readable):
		//
		//		tmp	= MatUtils.scalarMultiply(tmp, 1 - damping);
		//		R	= MatUtils.scalarMultiply(R, damping);
		//		R	= MatUtils.add(R, tmp);
		//
		// But it requires two extra MXM passes, which can be costly...
		// We know R & tmp are both m X m, so we can combine the 
		// three steps all together...
		// Finally, compute availability -- start by setting anything 
		// less than 0 to 0 in tmp. Also calc column sums in same pass...
		int ind = 0;
		final double omd = 1.0 - damping;
		
		for(int i = 0; i < m; i++) {
			// Get new max vector
			for(int j = 0; j < m; j++) 
				tmp[i][j] = S[i][j] - Y[i];
			tmp[ind][I[i]] = S[ind][I[i]] - Y2[ind++];
			
			// Perform damping, then piecewise 
			// calculate column sums
			for(int j = 0; j < m; j++) {
				tmp[i][j] *= omd;
				R[i][j] = (R[i][j] * damping) + tmp[i][j];

				tmp[i][j] = FastMath.max(R[i][j], 0);
				if(i != j) // Because we set diag after this outside j loop
					colSums[j] += tmp[i][j];
			}
			
			tmp[i][i] = R[i][i]; // Set diagonal elements in tmp equal to those in R
			colSums[i] += tmp[i][i];
		}
	}
	
	/**
	 * Computes the third portion of the AffinityPropagation iteration
	 * sequence in place. Separating this piece from the {@link #fit()} method
	 * itself allows for easier testing.
	 * @param tmp
	 * @param colSums
	 * @param A
	 * @param R
	 * @param mask
	 * @param damping
	 */
	protected static void affinityPiece3(double[][] tmp, double[] colSums, 
			double[][] A, double[][] R, double[] mask, double damping) {
		final int m = A.length;
		
		// Set any negative values to zero but keep diagonal at original
		// Originally ran this way, but costs an extra M x M operation:
		// tmp = MatUtils.scalarSubtract(tmp, colSums, Axis.COL);
		// Finally, more damping...
		// More damping ====================================
		// This can be done like this (which is more readable):
		//
		//		tmp	= MatUtils.scalarMultiply(tmp, 1 - damping);
		//		A	= MatUtils.scalarMultiply(A, damping);
		//		A	= MatUtils.subtract(A, tmp);
		//
		// But it requires two extra MXM passes, which can be costly... O(2M^2)
		// We know A & tmp are both m X m, so we can combine the 
		// three steps all together...
		
		// ALSO CHECK CONVERGENCE CRITERIA
		
		// Check convergence criteria =====================
		// This can be done like this for readability:
		//
		//		final double[] diagA = MatUtils.diagFromSquare(A);
		//		final double[] diagR = MatUtils.diagFromSquare(R);
		//		final double[] mask = new double[diagA.length];
		//		for(int i = 0; i < mask.length; i++)
		//			mask[i] = diagA[i] + diagR[i] > 0 ? 1d : 0d;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < m; j++) {
				tmp[i][j] -= colSums[j];
				
				if(tmp[i][j] < 0 && i != j) // Don't set diag to 0
					tmp[i][j] = 0;
				
				tmp[i][j] *= (1 - damping);
				A[i][j] = (A[i][j] * damping) - tmp[i][j];
			}
			
			mask[i] = A[i][i] + R[i][i] > 0 ? 1.0 : 0.0;
		}
	}
	
	
	@Override
	protected AffinityPropagation fit() {
		synchronized(fitLock) {
			if(null != labels)
				return this;
			
			
			
			// Init labels
			final LogTimer timer = new LogTimer();
			labels = new int[m];
			
			/*
			 * All elements singular
			 */
			if(this.singular_value) {
				warn("algorithm converged immediately due to all elements being equal in input matrix");
				this.converged = true;
				this.fitSummary.add(new Object[]{
					0,converged,timer.formatTime(),timer.formatTime(),1,timer.wallMsg()
				});
				
				sayBye(timer);
				return this;
			}
			
			
			sim_mat = computeSmoothedSimilarity(data.getData(), getSeparabilityMetric(), getSeed(), addNoise);
			info("computed similarity matrix and smoothed degeneracies in " + timer.toString());
			
			
			// Affinity propagation uses two matrices: the responsibility 
			// matrix, R, and the availability matrix, A
			double[][] A = new double[m][m];
			double[][] R = new double[m][m];
			double[][] tmp = new double[m][m]; // Intermediate staging...
			
			
			// Begin here
			int[] I = new int[m];
			double[][] e = new double[m][iterBreak];
			double[] Y;		// vector of arg maxes
			double[] Y2;	// vector of maxes post neg inf
			double[] sum_e;
			
			
			final LogTimer iterTimer = new LogTimer();
			info("beginning affinity computations " + timer.wallMsg());
			
			
			
			long iterStart = Long.MAX_VALUE;
			for(iterCt = 0; iterCt < maxIter; iterCt++) {
				iterStart = iterTimer.now();
				
				/*
				 * First piece in place
				 */
				Y = new double[m];
				Y2 = new double[m]; // Second max for each row
				affinityPiece1(A, sim_mat, tmp, I, Y, Y2);
				
				
				/*
				 * Second piece in place
				 */
				final double[] columnSums = new double[m];
				affinityPiece2(columnSums, tmp, I, sim_mat, R, Y, Y2, damping);
				
				
				/*
				 * Third piece in place
				 */
				final double[] mask = new double[m];
				affinityPiece3(tmp, columnSums, A, R, mask, damping);
					
					
				// Set the mask in `e`
				MatUtils.setColumnInPlace(e, iterCt % iterBreak, mask);
				numClusters = (int)VecUtils.sum(mask);
				
				
				
				if(iterCt >= iterBreak) { // Time to check convergence criteria...
					sum_e = MatUtils.rowSums(e);
					
					// masking
					int maskCt = 0;
					for(int i = 0; i < sum_e.length; i++)
						maskCt += sum_e[i] == 0 || sum_e[i] == iterBreak ? 1 : 0;
					
					converged = maskCt == m;
					
					if((converged && numClusters > 0) || iterCt == maxIter) {
						info("converged after " + (iterCt) + " iteration"+(iterCt!=1?"s":"") + 
							" in " + iterTimer.toString());
						break;
					} // Else did not converge...
				} // End outer if
				
				
				fitSummary.add(new Object[]{
					iterCt, converged, 
					iterTimer.formatTime( iterTimer.now() - iterStart ),
					timer.formatTime(),
					numClusters,
					timer.wallTime()
				});
			} // End for

			
			
			if(!converged) warn("algorithm did not converge");
			else { // needs one last info
				fitSummary.add(new Object[]{
					iterCt, converged, 
					iterTimer.formatTime( iterTimer.now() - iterStart ),
					timer.formatTime(),
					numClusters,
					timer.wallTime()
				});
			}
			
			
			info("labeling clusters from availability and responsibility matrices");
			
			
			// sklearn line: I = np.where(np.diag(A + R) > 0)[0]
			final ArrayList<Integer> arWhereOver0 = new ArrayList<>();
			
			// Get diagonal of A + R and add to arWhereOver0 if > 0
			// Could do this: MatUtils.diagFromSquare(MatUtils.add(A, R));
			// But takes 3M time... this takes M
			for(int i = 0; i < m; i++)
				if(A[i][i] + R[i][i] > 0)
					arWhereOver0.add(i);
			
			// Reassign to array, so whole thing takes 1M + K rather than 3M + K
			I = new int[arWhereOver0.size()];
			for(int j = 0; j < I.length; j++) I[j] = arWhereOver0.get(j);
			
			
			
			
			// Assign final K -- sklearn line: K = I.size  # Identify exemplars
			numClusters = I.length;
			info(numClusters+" cluster" + (numClusters!=1?"s":"") + " identified");
			
			
			
			// Assign the labels
			if(numClusters > 0) {
				
				/*
				 * I holds the columns we want out of sim_mat,
				 * retrieve this cols, do a row-wise argmax to get 'c'
				 * sklearn line: c = np.argmax(S[:, I], axis=1)
				 */
				double[][] over0cols = new double[m][numClusters];
				int over_idx = 0;
				for(int i: I)
					MatUtils.setColumnInPlace(over0cols, over_idx++, MatUtils.getColumn(sim_mat, i));

				
				
				/*
				 * Identify clusters
				 * sklearn line: c[I] = np.arange(K)  # Identify clusters
				 */
				int[] c = MatUtils.argMax(over0cols, Axis.ROW);
				int k = 0;
				for(int i: I)
					c[i] = k++;
				
				
				/* Refine the final set of exemplars and clusters and return results
				 * sklearn:
				 * 
				 *  for k in range(K):
			     *      ii = np.where(c == k)[0]
			     *      j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
			     *      I[k] = ii[j]
				 */
				ArrayList<Integer> ii = null;
				int[] iii = null;
				for(k = 0; k < numClusters; k++) {
					// indices where c == k; sklearn line: 
					// ii = np.where(c == k)[0]
					ii = new ArrayList<Integer>();
					for(int u = 0; u < c.length; u++)
						if(c[u] == k)
							ii.add(u);
					
					// Big block to break down sklearn process
					// overall sklearn line: j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
					iii = new int[ii.size()]; // convert to int array for MatUtils
					for(int j = 0; j < iii.length; j++) iii[j] = ii.get(j);
					
					
					// sklearn line: S[ii[:, np.newaxis], ii]
					double[][] cube = MatUtils.getRows(MatUtils.getColumns(sim_mat, iii), iii);
					double[] colSums = MatUtils.colSums(cube);
					final int argMax = VecUtils.argMax(colSums);
					
					
					// sklearn: I[k] = ii[j]
					I[k] = iii[argMax];
				}
				
				
				// sklearn line: c = np.argmax(S[:, I], axis=1)
				double[][] colCube = MatUtils.getColumns(sim_mat, I);
				c = MatUtils.argMax(colCube, Axis.ROW);
				
				
				// sklearn line: c[I] = np.arange(K)
				for(int j = 0; j < I.length; j++) // I.length == K, == numClusters
					c[I[j]] = j;
				
				
				// sklearn line: labels = I[c]
				for(int j = 0; j < m; j++)
					labels[j] = I[c[j]];
				
				
				/* 
				 * Reduce labels to a sorted, gapless, list
				 * sklearn line: cluster_centers_indices = np.unique(labels)
				 */
				centroidIndices = new ArrayList<Integer>(numClusters);
				for(Integer i: labels) // force autobox
					if(!centroidIndices.contains(i)) // Not race condition because synchronized
						centroidIndices.add(i);
				
				/*
				 * final label assignment...
				 * sklearn line: labels = np.searchsorted(cluster_centers_indices, labels)
				 */
				for(int i = 0; i < labels.length; i++)
					labels[i] = centroidIndices.indexOf(labels[i]);
				
				/*
				 * Don't forget to assign the centroids!
				 */
				this.centroids = new ArrayList<>();
				for(Integer idx: centroidIndices) {
					this.centroids.add(this.data.getRow(idx));
				}
			} else {
				centroids = new ArrayList<>(); // Empty
				centroidIndices = new ArrayList<>(); // Empty
				for(int i = 0; i < m; i++)
					labels[i] = -1; // Missing
			}

			
			// Clean up
			sim_mat = null;
			
			// Since cachedA/R are volatile, it's more expensive to make potentially hundreds(+)
			// of writes to a volatile class member. To save this time, reassign A/R only once.
			cachedA = A;
			cachedR = R;				
			
			sayBye(timer);
			
			return this;
		}
		
	} // End fit
	
	@Override
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","Iter. Time","Tot. Time","Num Clusters","Wall"
		};
	}

	@Override
	public ArrayList<double[]> getCentroids() {
		if(null == centroids)
			error(new ModelNotFitException("model has not yet been fit"));
		
		final ArrayList<double[]> cent = new ArrayList<double[]>();
		for(double[] d : centroids)
			cent.add(VecUtils.copy(d));
		
		return cent;
	}
	
	/** {@inheritDoc} */
	@Override
	public int[] predict(RealMatrix newData) {
		return CentroidUtils.predict(this, newData);
	}
}
