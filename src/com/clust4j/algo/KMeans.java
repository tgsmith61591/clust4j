package com.clust4j.algo;

import java.util.ArrayList;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class KMeans extends AbstractKCentroidClusterer {
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	
	public KMeans(final AbstractRealMatrix data, final int k) {
		this(data, new BaseKCentroidPlanner(k));
	}
	
	public KMeans(final AbstractRealMatrix data, final BaseKCentroidPlanner builder) {
		super(data, builder);
	}
	
	

	@Override
	final protected TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabels() {
		/* Key is the closest centroid, value is the records that belong to it */
		TreeMap<Integer, ArrayList<Integer>> cent = new TreeMap<Integer, ArrayList<Integer>>();
		
		/* Loop over each record in the matrix */
		for(int rec = 0; rec < m; rec++) {
			final double[] record = data.getRow(rec);
			int closest_cent = predict(record);
			
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
	final public void train() {
		if(isTrained)
			return;
		
		trainDistance();
		// TODO: if is similarity, do a different procedure...
	} // End train
	
	
	/**
	 * The training procedure for a distance-based GeometricallySeparable metric
	 */
	private void trainDistance() {
		if(verbose) info("beginning training segmentation for K = " + k);
			
		
		Double oldCost = null;
		labels = new int[m];
		
		// Enclose in for loop to ensure completes in proper iterations
		for(iter = 0; iter < maxIter; iter++) {
			
			
			if(verbose && iter%10 == 0)  {
				info("training iteration " + iter +
						"; current system cost = " + 
						oldCost ); //+ "; " + centroidsToString());
			}
			
			
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
					if(verbose)
						info("training reached convergence at iteration "+ iter + 
								"; Total system cost: " + cost);
					
					isTrained = true;
					converged = true;
					iter++; // Track iters used
					return;
				} else {
					oldCost = newCost;
				}
			}
		} // End iter for
		
		
		if(verbose)
			warn("algorithm did not converge. Total system cost: " + cost);
		
		// If the SSE delta never converges, still need to set isTrained to true
		isTrained = true;
	}
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEANS;
	}
}
