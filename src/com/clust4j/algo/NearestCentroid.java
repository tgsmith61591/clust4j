package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.SupervisedEvaluationMetric;
import com.clust4j.utils.Distance;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

/**
 * A supervised clustering algorithm used to predict a record's membership 
 * within a series of centroids. Note that this class implicitly utilizes
 * {@link LabelEncoder}, and will throw an {@link IllegalArgumentException} for
 * instances where the labels are of a single class.
 * @author Taylor G Smith
 */
public class NearestCentroid extends AbstractClusterer implements SupervisedClassifier, CentroidLearner {
	private static final long serialVersionUID = 8136673281643080951L;
	
	private Double shrinkage = null;
	private final int[] y_truth;
	private final int[] y_encodings;
	private final int m;
	private final int numClasses;
	private final LabelEncoder encoder;
	
	
	// State set in fit method
	volatile private int[] labels = null;
	volatile private ArrayList<double[]> centroids = null;
	
	/**
	 * Default constructor. Builds an instance of {@link NearestCentroid}
	 * with the default {@link NearestCentroidPlanner}
	 * @param data
	 * @param y
	 * @throws DimensionMismatchException if the dims of y do not match the dims of data
	 * @throws IllegalArgumentException if there is only one unique class in y
	 */
	public NearestCentroid(AbstractRealMatrix data, int[] y) {
		this(data, y, new NearestCentroidPlanner());
	}
	
	/**
	 * Builds an instance of {@link NearestCentroid}
	 * with an existing instance of {@link NearestCentroidPlanner}
	 * @param data
	 * @param y
	 * @param planner
	 * @throws DimensionMismatchException if the dims of y do not match the dims of data
	 * @throws IllegalArgumentException if there is only one unique class in y
	 */
	public NearestCentroid(AbstractRealMatrix data, int[] y, NearestCentroidPlanner planner) {
		super(data, planner);

		String err;
		VecUtils.checkDims(y);
		if((m=data.getRowDimension()) != y.length) {
			err = "mismatch in label dimensions and row dimension of data";
			error(err);
			throw new DimensionMismatchException(y.length, m);
		}
		
		
		// Build the label encoder
		try {
			this.encoder = new LabelEncoder(y).fit();
		} catch(IllegalArgumentException e) {
			error(e.getMessage());
			throw new IllegalArgumentException("Error in NearestCentroid: " + e.getMessage(), e);
		}
		
		
		this.numClasses = encoder.numClasses;
		this.y_truth = VecUtils.copy(y);
		this.y_encodings = encoder.getEncodedLabels();
		
		/*
		if(!(planner.getSep() instanceof DistanceMetric)) {
			err = "only distance metrics permitted for NearestCentroid; "
				+ "falling back to default: " + DEF_DIST;
			warn(err);
			super.setSeparabilityMetric(DEF_DIST);
		}
		*/
		
		this.shrinkage = planner.shrinkage;
		meta("shrinkage param="+shrinkage);
		meta("num classes="+numClasses);
	}
	
	
	public static class NearestCentroidPlanner 
			extends BaseClustererPlanner 
			implements SupervisedClassifierPlanner {
		
		private FeatureNormalization norm = DEF_NORMALIZER;
		private GeometricallySeparable met= DEF_DIST;
		private Double shrinkage = null;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		
		
		public NearestCentroidPlanner() { }

		@Override
		public NearestCentroid buildNewModelInstance(AbstractRealMatrix data, int[] y) {
			return new NearestCentroid(data, y, copy());
		}

		@Override
		public NearestCentroidPlanner copy() {
			return new NearestCentroidPlanner()
				.setNormalizer(norm)
				.setScale(scale)
				.setSeed(seed)
				.setSep(met)
				.setShrinkage(shrinkage)
				.setVerbose(verbose);
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public GeometricallySeparable getSep() {
			return met;
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
		public NearestCentroidPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}

		@Override
		public NearestCentroidPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}

		@Override
		public NearestCentroidPlanner setSeed(Random rand) {
			this.seed = rand;
			return this;
		}
		
		public NearestCentroidPlanner setShrinkage(final Double d) {
			this.shrinkage = d;
			return this;
		}

		@Override
		public NearestCentroidPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}

		@Override
		public NearestCentroidPlanner setSep(GeometricallySeparable dist) {
			this.met = dist;
			return this;
		}
		
	}


	@Override
	public ArrayList<double[]> getCentroids() {
		try {
			ArrayList<double[]> out= new ArrayList<>();
			for(double[] centroid: centroids)
				out.add(VecUtils.copy(centroid));
			
			return out;
		} catch(NullPointerException n) {
			throw new ModelNotFitException("model not yet fit", n);
		}
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.NEAREST;
	}

	/**
	 * Returns the labels predicted during the fitting method.
	 * To get the original truth set of training labels, use
	 * {@link #getTrainingLabels()}
	 */
	@Override
	public int[] getLabels() {
		try {
			return VecUtils.copy(labels);
		} catch(NullPointerException n) {
			throw new ModelNotFitException("model has not yet been fit", n);
		}
	}

	@Override
	public String getName() {
		return "NearestCentroid";
	}

	/**
	 * Returns a copy of the training labels
	 * (the truth set)
	 */
	@Override
	public int[] getTrainingLabels() {
		return VecUtils.copy(y_truth);
	}

	@Override
	public NearestCentroid fit() {
		synchronized(this) {
			
			try {
				if(null != labels) // already fit
					return this;
				
				
				final LogTimer timer = new LogTimer();
				this.centroids = new ArrayList<double[]>(numClasses);
				final int[] nk = new int[numClasses]; // the count of clusters in each class
				
				final boolean isManhattan = getSeparabilityMetric()
					.equals(Distance.MANHATTAN);
				
				boolean[] mask;
				double[][] masked;
				double[] centroid;
				
				info("identifying centroid for each class label");
				for(int currentClass = 0; currentClass < numClasses; currentClass++) {
					// Since we've already encoded the labels, we can just use
					// an iterator like this to keep track of the current one
					
					mask = new boolean[m];
					for(int j = 0; j < m; j++)
						mask[j] = y_encodings[j] == currentClass;
					nk[currentClass] = VecUtils.sum(mask);
					
					
					masked = new double[nk[currentClass]][];
					for(int j = 0, k = 0; j < m; j++)
						if(mask[j])
							masked[k++] = data.getRow(j);
					
					
					// Update
					centroid = isManhattan ? MatUtils.medianRecord(masked) : MatUtils.meanRecord(masked);
					centroids.add(centroid);
				}
				
				
				if(null != shrinkage) {
					info("applying smoothing to class centroids");
					double[][] X = data.getData();
					centroid = MatUtils.meanRecord(X);
					
					// determine deviation
					double[] em = getMVec(nk, m);
					double[] variance = variance(X, centroids, y_encodings);
					double[] s = sqrtMedAdd(variance, m, numClasses);
					double[][] ms = mmsOuterProd(em, s);
					double[][] shrunk = getDeviationMinShrink(centroids, centroid, ms, shrinkage);
					
					for(int i = 0; i < numClasses; i++)
						for(int j = 0; j < centroid.length; j++)
							centroids.get(i)[j] = shrunk[i][j] + centroid[j];
				}
				
				
				// Now run the predict method on training labels to score model
				this.labels = predict(data);
				info("model score ("+DEF_SUPERVISED_METRIC+"): " + score());
				
				
				sayBye(timer);
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
		} // end synch
	}
	
	// Tested: passing
	static double[][] getDeviationMinShrink(ArrayList<double[]> centroids, 
				double[] centroid, double[][] ms, double shrinkage) {
		final int m = centroids.size(), n = centroid.length;
		
		double[] cent;
		final double[][] dev = new double[m][n];
		for(int i = 0; i < m; i++) {
			cent = centroids.get(i);
			
			int sign = 1;
			for(int j = 0; j < n; j++) {
				double val = (cent[j] - centroid[j]) / ms[i][j];
				sign = val > 0 ? 1 : -1;
				dev[i][j] = ms[i][j] * sign 
					* FastMath.max(0, FastMath.abs(val) - shrinkage);
			}
		}
		
		return dev;
	}
	
	// Tested: passing
	static double[] getMVec(int[] nk, int m) {
		double[] em = new double[nk.length];
		for(int i = 0; i < em.length; i++)
			em[i] = FastMath.sqrt((1.0/nk[i]) + (1.0/m));
		
		return em;
	}
	
	// Tested: passing
	static double[][] mmsOuterProd(double[] m, double[] s) {
		return VecUtils.outerProduct(m, s);
	}
	
	// Tested: passing
	static double[] sqrtMedAdd(double[] variance, int m, int numClasses) {
		double[] s = new double[variance.length];
		double m_min_n = (double)(m - numClasses);
		
		for(int i = 0; i < s.length; i++)
			s[i] = FastMath.sqrt(variance[i] / m_min_n);
		final double s_med = VecUtils.median(s);
		for(int i = 0; i < s.length; i++)
			s[i] += s_med;
		return s;
	}
	
	// Tested: passing
	static double[] variance(double[][] X, ArrayList<double[]> centroids, int[] y_ind) {
		int m = X.length, n = X[0].length;
		
		// sklearn line:
		// variance = (X - self.centroids_[y_ind]) ** 2
        // variance = variance.sum(axis=0)
		// Get the column sums of X - centroid (row wise) times itself
		// (each element squared)
		double val;
		double[] variance = new double[n], centroid;
		for(int i = 0; i < m; i++) {
			centroid = centroids.get(y_ind[i]);
			
			for(int j = 0; j < n; j++) {
				val = X[i][j] - centroid[j];
				variance[j] += (val * val);
			}
		}
		
		return variance;
	}

	@Override
	public double score() {
		return score(BaseClassifier.DEF_SUPERVISED_METRIC);
	}
	
	@Override
	public double score(SupervisedEvaluationMetric metric) {
		final int[] predicted = getLabels(); // Propagates a model not fit exception if not fit...
		return metric.evaluate(y_truth, predicted);
	}

	@Override
	public int[] predict(AbstractRealMatrix newData) {
		return predict(newData.getData()).getKey();
	}
	
	/**
	 * To be used from {@link KMeans}
	 * @param data
	 * @return
	 */
	protected EntryPair<int[], double[]> predict(double[][] data) {
		if(null == centroids)
			throw new ModelNotFitException("model not yet fit");
		
		int[] predictions = new int[data.length];
		double[] dists = new double[data.length];
		double[] row, centroid;
		
		for(int i = 0; i < data.length; i++) {
			row = data[i];
			
			double minDist = Double.POSITIVE_INFINITY, dist;
			int nearestLabel = -1;
			
			for(int j = 0; j < centroids.size(); j++) {
				centroid = centroids.get(j);
				dist = getSeparabilityMetric().getDistance(centroid, row);
				
				if(dist < minDist) {
					minDist = dist;
					nearestLabel = j;
				}
			}
			
			predictions[i] = nearestLabel;
			dists[i] = minDist;
		}
		
		return new EntryPair<>(encoder.reverseTransform(predictions), dists);
	}
}
