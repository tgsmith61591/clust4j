package com.clust4j.algo;

import java.util.ArrayList;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class KMeans extends AbstractKCentroidClusterer {
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final public static double DEF_MIN_CHNG = 0.005;
	
	final private double minChange;
	
	
	
	final public static class KMeansPlanner extends BaseKCentroidPlanner {
		private double minChange = DEF_MIN_CHNG;

		public KMeansPlanner(int k) {
			super(k);
		}

		public KMeansPlanner setMinChangeStoppingCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public KMeansPlanner setDist(final GeometricallySeparable dist) {
			return (KMeansPlanner) super.setDist(dist);
		}
		
		@Override
		public KMeansPlanner setMaxIter(final int max) {
			return (KMeansPlanner) super.setMaxIter(max);
		}
		
		public KMeansPlanner setScale(final boolean scale) {
			return (KMeansPlanner) super.setScale(scale);
		}
	}
	
	
	
	
	public KMeans(final AbstractRealMatrix data, final int k) {
		this(data, new KMeansPlanner(k));
	}
	
	public KMeans(final AbstractRealMatrix data, final KMeansPlanner builder) {
		super(data, builder);
		this.minChange = builder.minChange;
	}
	
	
	
	
	/**
	 * Calculates the SSE within the provided cluster
	 * @param inCluster
	 * @return Sum of Squared Errors
	 */
	final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
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
	
	public double getMinChange() {
		return minChange;
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

	private TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabels() {
		/* Key is the closest centroid, value is the records that belong to it */

		TreeMap<Integer, ArrayList<Integer>> cent = new TreeMap<Integer, ArrayList<Integer>>();
		final int m = data.getRowDimension();
		
		/* Loop over each record in the matrix */
		for(int rec = 0; rec < m; rec++) {
			final double[] record = data.getRow(rec);
			double min_dist = Double.MAX_VALUE;
			int closest_cent = 0;
			
			/* Loop over every centroid, get calculate dist from record,
			 * identify the closest centroid to this record */
			for(int i = 0; i < k; i++) {
				final double[] centroid = centroids.get(i);
				final double dis = getDistanceMetric().distance(record, centroid);
				
				/* Track the current min distance. If dist
				 * is shorter than the previous min, assign
				 * new closest centroid to this record */
				if(dis < min_dist) {
					min_dist = dis;
					closest_cent = i;
				}
			}
			
			labels[rec] = closest_cent;
			if(cent.get(closest_cent) == null)
				cent.put(closest_cent, new ArrayList<Integer>());
			
			cent.get(closest_cent).add(rec);
		}
		
		return cent;
	}
	
	@Override
	final public void train() {
		if(isTrained)
			return;
		
		//initCentroids(); Now initialized in super
		Double oldCost = null;
		labels = new int[m];
		
		// Enclose in for loop to ensure completes in proper iterations
		for(iter = 0; iter < maxIter; iter++) {
			
			/* Key is the closest centroid, value is the records that belong to it */
			cent_to_record = assignClustersAndLabels();
			
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
				// Evaluate new SSE vs old SSE. If meets stopping criteria, break,
				// otherwise update new SSE and continue.
				if( FastMath.abs(oldCost - newCost) < minChange ) {
					isTrained = true;
					converged = true;
					iter++; // Track iters used
					return;
				} else {
					oldCost = newCost;
				}
			}
		} // End iter for
		
		
		// If the SSE delta never converges, still need to set isTrained to true
		isTrained = true;
	} // End train
}
