package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

/**
 * <a href="https://en.wikipedia.org/wiki/K-means_clustering">KMeans clustering</a> is
 * a method of vector quantization, originally from signal processing, that is popular 
 * for cluster analysis in data mining. KMeans clustering aims to partition <i>m</i> 
 * observations into <i>k</i> clusters in which each observation belongs to the cluster 
 * with the nearest mean, serving as a prototype of the cluster. This results in 
 * a partitioning of the data space into <a href="https://en.wikipedia.org/wiki/Voronoi_cell">Voronoi cells</a>.
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public class KMeans extends AbstractCentroidClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1102324012006818767L;
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final public static int DEF_MAX_ITER = 100;
	
	
	
	public KMeans(final AbstractRealMatrix data) {
		this(data, DEF_K);
	}
	
	public KMeans(final AbstractRealMatrix data, final int k) {
		this(data, new KMeansPlanner(k));
	}
	
	public KMeans(final AbstractRealMatrix data, final KMeansPlanner planner) {
		super(data, planner);
	}
	
	
	public static class KMeansPlanner extends CentroidClustererPlanner {
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_MIN_CHNG;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private int k = DEF_K;
		
		public KMeansPlanner() { }
		public KMeansPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public KMeans buildNewModelInstance(final AbstractRealMatrix data) {
			return new KMeans(data, this.copy());
		}
		
		@Override
		public KMeansPlanner copy() {
			return new KMeansPlanner(k)
				.setMaxIter(maxIter)
				.setMinChangeStoppingCriteria(minChange)
				.setScale(scale)
				.setSep(dist)
				.setVerbose(verbose)
				.setSeed(seed)
				.setNormalizer(norm);
		}
		
		@Override
		public int getK() {
			return k;
		}
		
		@Override
		public int getMaxIter() {
			return maxIter;
		}
		
		@Override
		public double getMinChange() {
			return minChange;
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
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public KMeansPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public KMeansPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}

		public KMeansPlanner setMinChangeStoppingCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public KMeansPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public KMeansPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public KMeansPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public KMeansPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	
	
	final TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabelsInPlace() {
		/* Key is the closest centroid, value is the records that belong to it */
		TreeMap<Integer, ArrayList<Integer>> cent = new TreeMap<Integer, ArrayList<Integer>>();
		
		/* Loop over each record in the matrix */
		for(int rec = 0; rec < m; rec++) {
			final double[] record = data.getRow(rec);
			int closest_cent = predictCentroid(record);
			
			labels[rec] = closest_cent;
			if(cent.get(closest_cent) == null)
				cent.put(closest_cent, new ArrayList<Integer>());
			
			cent.get(closest_cent).add(rec);
		}
		
		return cent;
	}
	
	
	/**
	 * Calculates the SSE within the provided cluster
	 * @param inCluster
	 * @return Sum of Squared Errors
	 */
	private final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
		// Now calc the SSE in cluster i
		double sumI = 0;
		final int n = newCentroid.length;
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			for(int j = 0; j < n; j++) {
				final double diff = record[j] - newCentroid[j];
				sumI += diff * diff;
			}
		}
		
		return sumI;
	}
	
	@Override
	public String getName() {
		return "KMeans";
	}

	private double[] idNewCentroid(ArrayList<Integer> inCluster) {
		final int n = data.getColumnDimension();
		final double[] newCentroid = new double[n];
		
		// Put col sums of belonging records in newCentroid
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			for(int j = 0; j < n; j++)
				newCentroid[j] += record[j];
		}
		
		// Set newCentroid to means
		for(int j = 0; j < n; j++)
			newCentroid[j] /= (double) inCluster.size();
		
		return newCentroid;
	}
	
	@Override
	final public KMeans fit() {
		synchronized(this) { // Must be synchronized because alters internal structs
			
			try {
				if(null!=labels) // Already have fit this model
					return this;
				
	
				final long start = System.currentTimeMillis();
				info("beginning training segmentation for K = " + k);
					
				
				Double oldCost = null;
				labels = new int[m];
				
				// Enclose in for loop to ensure completes in proper iterations
				long iterStart = System.currentTimeMillis();
				
				
				OuterLoop:
				for(iter = 0; iter < maxIter; iter++) {
					
					
					if(iter%10 == 0)  {
						info("training iteration " + iter +
								"; current system cost = " + 
								oldCost ); //+ "; " + centroidsToString());
					}
					
					
					/* Key is the closest centroid, value is the records that belong to it */
					cent_to_record = assignClustersAndLabelsInPlace();
						
					
					
					// Now reassign centroids based on records inside cluster
					ArrayList<double[]> newCentroids = new ArrayList<double[]>();
					double newCost = 0;
					
					/* Iterate over each centroid, calculate barycentric mean of
					 * the points that belong in that cluster as the new centroid */
					for(int i = 0; i < k; i++) {
						
						/* The record numbers that belong to this cluster */
						final ArrayList<Integer> inCluster = cent_to_record.get(i);
						final double[] newCentroid = idNewCentroid(inCluster);
						newCentroids.add(newCentroid);
						newCost += getCost(inCluster, newCentroid);
					}
					
					
					// move current newSSE to oldSSE, check stopping condition...
					centroids = newCentroids;
					cost = newCost;
					
					if(null == oldCost) { // First iteration
						oldCost = newCost;
					} else { // At least second iteration, can check delta
						// Evaluate new SSE vs. old SSE. If meets stopping criteria, break,
						// otherwise update new SSE and continue.
						if( FastMath.abs(oldCost - newCost) < minChange ) {
							info("training reached convergence at iteration "+ iter + " (avg iteration time: " + 
								LogTimeFormatter.millis( (long) ((long)(System.currentTimeMillis()-iterStart)/
									(double)(iter+1)), false) + ")");
							
							converged = true;
							iter++; // Track iters used
							
							break OuterLoop;
						} else {
							oldCost = newCost;
						}
					}
				} // End iter for
				
				
				info("Total system cost: " + cost);
				if(!converged) warn("algorithm did not converge");
				wrapItUp(start);
				
				
				reorderLabels();
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synchronized
		
	} // End train
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEANS;
	}
	
	private int predictCentroid(final double[] newRecord) {
		int nearestLabel = 0;
		double shortestDist = Double.MAX_VALUE;
		double[] cent;
		for(int i = 0; i < k; i++) {
			cent = centroids.get(i);
			double dist = getSeparabilityMetric().getDistance(newRecord, cent);
			
			if(dist < shortestDist) {
				shortestDist = dist;
				nearestLabel = i;
			}
		}

		return nearestLabel;
	}
	
	final private void reorderLabels() {
		// Now rearrange labels in order... first get unique labels in order of appearance
		final ArrayList<Integer> orderOfLabels = new ArrayList<Integer>(k);
		for(int label: labels) {
			if(!orderOfLabels.contains(label)) // Race condition? but synchronized so should be ok...
				orderOfLabels.add(label);
		}
		
		final int[] newLabels = new int[m];
		final TreeMap<Integer, double[]> newCentroids = new TreeMap<>();
		for(int i = 0; i < m; i++) {
			final Integer idx = orderOfLabels.indexOf(labels[i]);
			newLabels[i] = idx;
			
			if(!newCentroids.containsKey(idx))
				newCentroids.put(idx, centroids.get(labels[i]));
		}
		
		// Reassign labels...
		labels = newLabels;
		cent_to_record = null;
		centroids = new ArrayList<>(newCentroids.values());
	}

	public double totalCost() {
		return cost;
	}
}
