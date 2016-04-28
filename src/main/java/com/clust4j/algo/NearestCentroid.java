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

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.CircularKernel;
import com.clust4j.kernel.LogKernel;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.utils.ArrayFormatter;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

/**
 * A supervised clustering algorithm used to predict a record's membership 
 * within a series of centroids. Note that this class implicitly utilizes
 * {@link LabelEncoder}, and will throw an {@link IllegalArgumentException} for
 * instances where the labels are of a single class.
 * @author Taylor G Smith
 */
final public class NearestCentroid extends AbstractClusterer implements SupervisedClassifier, CentroidLearner {
	private static final long serialVersionUID = 8136673281643080951L;
	final public static HashSet<Class<? extends GeometricallySeparable>> UNSUPPORTED_METRICS;
	
	
	/**
	 * Static initializer
	 */
	static {
		UNSUPPORTED_METRICS = new HashSet<>();
		UNSUPPORTED_METRICS.add(CircularKernel.class);
		UNSUPPORTED_METRICS.add(LogKernel.class);
		// Add metrics here if necessary...
	}
	
	@Override final public boolean isValidMetric(GeometricallySeparable geo) {
		return !UNSUPPORTED_METRICS.contains(geo.getClass());
	}
	
	
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
	 * with the default {@link NearestCentroidParameters}
	 * @param data
	 * @param y
	 * @throws DimensionMismatchException if the dims of y do not match the dims of data
	 * @throws IllegalArgumentException if there is only one unique class in y
	 */
	protected NearestCentroid(RealMatrix data, int[] y) {
		this(data, y, new NearestCentroidParameters());
	}
	
	/**
	 * Builds an instance of {@link NearestCentroid}
	 * with an existing instance of {@link NearestCentroidParameters}
	 * @param data
	 * @param y
	 * @param planner
	 * @throws DimensionMismatchException if the dims of y do not match the dims of data
	 * @throws IllegalArgumentException if there is only one unique class in y
	 */
	protected NearestCentroid(RealMatrix data, int[] y, NearestCentroidParameters planner) {
		super(data, planner);

		VecUtils.checkDims(y);
		if((m=data.getRowDimension()) != y.length)
			error(new DimensionMismatchException(y.length, m));
		
		// Build the label encoder
		/*
		try {
			this.encoder = new LabelEncoder(y).fit();
		} catch(IllegalArgumentException e) {
			error(e.getMessage());
			throw new IllegalArgumentException("Error in NearestCentroid: " + e.getMessage(), e);
		}
		*/
		
		// Opting for SafeLabelEncoder in favor of allowing single class systems...
		this.encoder = new SafeLabelEncoder(y).fit();
		
		
		this.numClasses = encoder.numClasses;
		this.y_truth = VecUtils.copy(y);
		this.y_encodings = encoder.getEncodedLabels();
		
		/*
		 * Check metric for validity
		 */
		if(!isValidMetric(this.dist_metric)) {
			warn(this.dist_metric.getName() + " is not valid for "+getName()+". "
				+ "Falling back to default Euclidean dist");
			setSeparabilityMetric(DEF_DIST);
		}
		
		this.shrinkage = planner.getShrinkage();
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
			"Num Rows","Num Cols","Metric","Num Classes",
			"Shrinkage","Allow Par."
		}, new Object[]{
			m,data.getColumnDimension(),getSeparabilityMetric(),numClasses,
			shrinkage,
			parallel
		});
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
		return super.handleLabelCopy(labels);
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
	protected NearestCentroid fit() {
		synchronized(fitLock) {
			
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
			
			int encoded;
			info("identifying centroid for each class label");
			for(int currentClass = 0; currentClass < numClasses; currentClass++) {
				// Since we've already encoded the labels, we can just use
				// an iterator like this to keep track of the current one
				encoded = encoder.reverseEncodeOrNull(currentClass); // shouldn't ever be null
				
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

				fitSummary.add(new Object[]{
					encoded, 
					nk[currentClass], 
					barycentricDistance(masked, centroid), 
					ArrayFormatter.arrayToString(centroid),
					timer.wallTime()
				});
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
		}
	}
	

	
	/**
	 * For computing the total sum of squares
	 * @param instances
	 * @param centroid
	 * @return
	 */
	protected static double barycentricDistance(double[][] instances, double[] centroid) {
		double clust_cost = 0.0, diff;
		final int n = centroid.length;
		
		for(double[] instance: instances) {
			/* internal method, so shouldn't happen...
			if(n != instance.length)
				throw new DimensionMismatchException(n, instance.length);
			*/
			
			for(int j = 0; j < n; j++) {
				diff = instance[j] - centroid[j];
				clust_cost += diff * diff;
			}
		}
		
		return clust_cost;
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
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Class Label","Num. Instances","WSS","Centroid","Wall"
		};
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
	public double score(SupervisedMetric metric) {
		final int[] predicted = getLabels(); // Propagates a model not fit exception if not fit...
		return metric.evaluate(y_truth, predicted);
	}

	@Override
	public int[] predict(RealMatrix newData) {
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
			
			double minDist = Double.POSITIVE_INFINITY, dist = minDist;
			int nearestLabel = 0; // should not equal -1, because dist could be infinity
			
			for(int j = 0; j < centroids.size(); j++) {
				centroid = centroids.get(j);
				dist = getSeparabilityMetric()
					.getPartialDistance(centroid, row); // Can afford to compute partial dist--faster
				
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
