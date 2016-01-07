package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.Convergeable;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.SimilarityMetric;
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
public class AffinityPropagation extends AbstractAutonomousClusterer implements Convergeable, Classifier, CentroidLearner {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1986169131867013043L;
	
	/** The number of stagnant iterations after which the algorithm will declare convergence */
	final public static int DEF_ITER_BREAK = 15;
	final public static int DEF_MAX_ITER = 200;
	final public static double DEF_MIN_CHANGE = 0d;
	final public static double DEF_DAMPING = 0.5;
	/** By default uses minute Gaussian smoothing. It is recommended this remain
	 *  true, but the {@link AffinityPropagationPlanner#useGaussianSmoothing(boolean)}
	 *  method can disable this option */
	final public static boolean DEF_ADD_GAUSSIAN_NOISE = true;
	
	
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
	private final double minChange;
	
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
	
	
	
	
	public AffinityPropagation(final AbstractRealMatrix data) {
		this(data, new AffinityPropagationPlanner());
	}
	
	public AffinityPropagation(final AbstractRealMatrix data, final AffinityPropagationPlanner planner) {
		super(data, planner);
		String error;
		
		
		// Check some args
		if(planner.damping < DEF_DAMPING || planner.damping >= 1) {
			error = "damping must be between " + DEF_DAMPING + " and 1";
			error(error);
			throw new IllegalArgumentException(error);
		}
		
		this.damping = planner.damping;
		this.iterBreak = planner.iterBreak;
		this.m = data.getRowDimension();
		this.minChange = planner.minChange;
		this.maxIter = planner.maxIter;
		this.addNoise = planner.addNoise;
		
		if(maxIter < 0)	throw new IllegalArgumentException("maxIter must exceed 0");
		if(minChange<0)	throw new IllegalArgumentException("minChange must exceed 0");
		if(iterBreak<0)	throw new IllegalArgumentException("iterBreak must exceed 0");
		
		
		meta("damping="+damping);
		meta("maxIter="+maxIter);
		meta("minChange="+minChange);
		meta("addNoise="+addNoise);
		
		if(!addNoise) {
			warn("not scaling with Gaussian noise can cause the algorithm not to converge");
		}
	}
	
	
	
	
	public static class AffinityPropagationPlanner extends AbstractClusterer.BaseClustererPlanner {
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_MIN_CHANGE;
		private int iterBreak = DEF_ITER_BREAK;
		
		private double damping = DEF_DAMPING;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private boolean addNoise = DEF_ADD_GAUSSIAN_NOISE;
		private FeatureNormalization norm = DEF_NORMALIZER;

		public AffinityPropagationPlanner() { /* Default constructor */ }
		
		public AffinityPropagationPlanner useGaussianSmoothing(boolean b) {
			this.addNoise = b;
			return this;
		}

		@Override
		public AffinityPropagation buildNewModelInstance(AbstractRealMatrix data) {
			return new AffinityPropagation(data, this);
		}
		
		@Override
		public AffinityPropagationPlanner copy() {
			return new AffinityPropagationPlanner()
				.setDampingFactor(damping)
				.setIterBreak(iterBreak)
				.setMaxIter(maxIter)
				.setMinChange(minChange)
				.setScale(scale)
				.setSeed(seed)
				.setSep(dist)
				.setVerbose(verbose)
				.useGaussianSmoothing(addNoise)
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
		
		public AffinityPropagationPlanner setDampingFactor(final double damp) {
			this.damping = damp;
			return this;
		}
		
		public AffinityPropagationPlanner setIterBreak(final int iters) {
			this.iterBreak = iters;
			return this;
		}
		
		public AffinityPropagationPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}
		
		public AffinityPropagationPlanner setMinChange(final double min) {
			this.minChange = min;
			return this;
		}

		@Override
		public AffinityPropagationPlanner setScale(boolean b) {
			scale = b;
			return this;
		}

		@Override
		public AffinityPropagationPlanner setSeed(Random rand) {
			seed = rand;
			return this;
		}

		@Override
		public AffinityPropagationPlanner setVerbose(boolean b) {
			verbose = b;
			return this;
		}

		@Override
		public AffinityPropagationPlanner setSep(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public AffinityPropagationPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
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
	public boolean didConverge() {
		return converged;
	}
	
	public double[][] getAvailabilityMatrix() {
		try {
			return MatUtils.copyMatrix(cachedA);
		} catch(NullPointerException npe) {
			throw new ModelNotFitException("model is not fit", npe);
		}
	}
	
	public double[][] getResponsibilityMatrix() {
		try {
			return MatUtils.copyMatrix(cachedR);
		} catch(NullPointerException npe) {
			throw new ModelNotFitException("model is not fit", npe);
		}
	}

	@Override
	public int getMaxIter() {
		return maxIter;
	}

	@Override
	public double getConvergenceTolerance() {
		return minChange;
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

	@Override
	public AffinityPropagation fit() {
		synchronized(this) {
			
			try {
				if(null != labels)
					return this;
				
				
				
				// Init labels
				final long start = System.currentTimeMillis();
				labels = new int[m];
				String error;
				
				
				
				// Calc sim mat to MAXIMIZE VALS
				final long sim_time = System.currentTimeMillis();
				if(getSeparabilityMetric() instanceof SimilarityMetric) {
					info("computing similarity matrix");
					sim_mat = ClustUtils.similarityFullMatrix(data, (SimilarityMetric)getSeparabilityMetric());
				} else {
					info("computing negative distance (pseudo similarity) matrix");
					sim_mat = MatUtils.negative(ClustUtils.distanceFullMatrix(data, getSeparabilityMetric()));
				}
				info("completed similarity computations in " + LogTimeFormatter.millis(System.currentTimeMillis()-sim_time, false));
				
				
				
				// Extract the upper triangular portion from sim mat, get the median as default pref 
				info("computing initialization point");
				int idx = 0, mChoose2 = ((m*m) - m) / 2;
				final double[] vals = new double[mChoose2];
				for(int i = 0; i < m - 1; i++)
					for(int j = i + 1; j < m; j++)
						vals[idx++] = sim_mat[i][j];
				
				final double pref = VecUtils.median(vals);
				info("pref = "+pref);
				
				
				// Place pref on diagonal of sim mat
				info("refactoring similarity matrix diagonal vector");
				for(int i = 0; i < m; i++)
					sim_mat[i][i] = pref;
				
				
				
				// Affinity propagation uses two matrices: the responsibility 
				// matrix, R, and the availability matrix, A
				double[][] A = new double[m][m];
				double[][] R = new double[m][m];
				double[][] tmp = new double[m][m]; // Intermediate staging...
				
				
				if(addNoise) {
					// Add some extremely small noise to the similarity matrix
					double[][] tiny_scaled = MatUtils.scalarMultiply(sim_mat, GlobalState.Mathematics.EPS);
					tiny_scaled = MatUtils.scalarAdd(tiny_scaled, GlobalState.Mathematics.TINY*100);
					
					info("removing matrix degeneracies; scaling with minute Gaussian noise");
					long gausStart = System.currentTimeMillis();
					double[][] noise = MatUtils.randomGaussian(m, m, getSeed());
					info("Gaussian noise matrix computed in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-gausStart, false));
					double[][] noiseMatrix = null;
					
					
					try {
						long multStart = System.currentTimeMillis();
						info("multiplying scaling matrix by noise matrix ("+m+"x"+m+")");
						noiseMatrix = MatUtils.multiply(tiny_scaled, noise);
						info("matrix product computed in " + 
							LogTimeFormatter.millis(System.currentTimeMillis()-multStart, false));
					} catch(DimensionMismatchException e) {
						error = e.getMessage();
						error(error);
						throw new InternalError("similarity matrix produced DimMismatch: "+ error); // Should NEVER happen
					}
					
					sim_mat = MatUtils.add(sim_mat, noiseMatrix);
				}
				
				
				// Begin here
				int[] I = new int[m];
				double[][] e = new double[m][iterBreak];
				double[] Y;		// vector of arg maxes
				double[] Y2;	// vector of maxes post neg inf
				double[] sum_e;
				
				long iterStart = System.currentTimeMillis();
				for(iterCt = 0; iterCt < maxIter; iterCt++) {
					
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
					Y = new double[m];
					Y2 = new double[m]; // Second max for each row
					for(int i = 0; i < m; i++) {
						double runningMax = GlobalState.Mathematics.SIGNED_MIN;
						double secondMax = GlobalState.Mathematics.SIGNED_MIN;
						int runningMaxIdx = -1;			// Idx of max row element
						for(int j = 0; j < m; j++) { 	// Create tmp as A + sim_mat
							tmp[i][j] = A[i][j] + sim_mat[i][j];
							
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
					final double[] columnSums = new double[m];
					for(int i = 0; i < m; i++) {
						// Get new max vector
						for(int j = 0; j < m; j++) tmp[i][j] = sim_mat[i][j] - Y[i];
						tmp[ind][I[i]] = sim_mat[ind][I[i]] - Y2[ind++];
						
						// Perform damping, then piecewise 
						// calculate column sums
						for(int j = 0; j < m; j++) {
							tmp[i][j] *= (1 - damping);
							R[i][j] = (R[i][j] * damping) + tmp[i][j];
	
							tmp[i][j] = FastMath.max(R[i][j], 0);
							if(i != j) // Because we set diag after this outside j loop
								columnSums[j] += tmp[i][j];
						}
						
						tmp[i][i] = R[i][i]; // Set diagonal elements in tmp equal to those in R
						columnSums[i] += tmp[i][i];
					}
					
					
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
					// But it requires two extra MXM passes, which can be costly...
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
					final double[] mask = new double[m];
					for(int i = 0; i < m; i++) {
						for(int j = 0; j < m; j++) {
							tmp[i][j] -= columnSums[j];
							
							if(tmp[i][j] < 0 && i != j) // Don't set diag to 0
								tmp[i][j] = 0;
							
							tmp[i][j] *= (1 - damping);
							A[i][j] = (A[i][j] * damping) - tmp[i][j];
						}
						
						mask[i] = A[i][i] + R[i][i] > 0 ? 1d : 0d;
					}
						
						
						
					// Set the mask in `e`
					MatUtils.setColumnInPlace(e, iterCt % iterBreak, mask);
					
					// Get k
					numClusters = (int)VecUtils.sum(mask);
	
					if(iterCt >= iterBreak) { // Time to check convergence criteria...
						sum_e = MatUtils.rowSums(e);
						
						// masking
						int maskCt = 0;
						for(int i = 0; i < sum_e.length; i++)
							maskCt += sum_e[i] == 0 || sum_e[i] == iterBreak ? 1 : 0;
						
						converged = maskCt == m;
						
						if((converged && numClusters > 0) || iterCt == maxIter) {
							iterCt++;
							info("converged after " + (iterCt) + " iteration"+(iterCt!=1?"s":"") + 
								" (avg iteration time: " + LogTimeFormatter.millis( (long) ((long)(System.currentTimeMillis()-iterStart)/(double)iterCt), false) + ")");
							break;
						} // Else did not converge...
					} // End outer if
				} // End for
	
				
				if(!converged) warn("algorithm did not converge");
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
					
				} else {
					centroids = new ArrayList<>(); // Empty
					centroidIndices = new ArrayList<>(); // Empty
					for(int i = 0; i < m; i++)
						labels[i] = -1; // Missing
				}
				
				
				info("model " + getKey() + " completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false) +
					System.lineSeparator());
				
				
				// Clean up
				sim_mat = null;
				
				// Since cachedA/R are volatile, it's more expensive to make potentially hundreds(+)
				// of writes to a volatile class member. To save this time, reassign A/R only once.
				cachedA = A;
				cachedR = R;
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synch
		
	} // End fit
	
	@Override
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}

	@Override
	public ArrayList<double[]> getCentroids() {
		return centroids;
	}
}
