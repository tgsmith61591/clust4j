package com.clust4j.algo;

import java.util.ArrayList;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class KMedoids extends AbstractKCentroidClusterer {
	final public static GeometricallySeparable DEF_DIST = Distance.MANHATTAN;
	
	final public static class KMedoidsPlanner extends BaseKCentroidPlanner {
		public KMedoidsPlanner(int k) {
			super(k);
			super.setDist(DEF_DIST);
		}
	}
	
	public KMedoids(final AbstractRealMatrix data, final int k) {
		this(data, new KMedoidsPlanner(k));
	}
	
	public KMedoids(final AbstractRealMatrix data, final BaseKCentroidPlanner builder) {
		super(data, builder);
	}
	
	/**
	 * Calculates the intracluster cost
	 * @param inCluster
	 * @return the sum of manhattan distances between vectors and the centroid
	 */
	@Override
	final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
		// Now calc the dissimilarity in the cluster
		double sumI = 0;
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			sumI += KMedoids.DEF_DIST.distance(record, newCentroid);
		}
		
		return sumI;
	}
	
	@Override
	public String getName() {
		return "KMedoids";
	}

	@Override
	public void train() {
		if(isTrained)
			return;

		
		// Initialize labels, initialize clusters
		labels = new int[m];
		cent_to_record = assignClustersAndLabels();
		
		
		// These variables hold the "best" so far (which minimize cost)
		Double oldCost = getCostOfSystem(); // Initial cost of system
		cost = oldCost;
		ArrayList<double[]> bestMedoids = centroids; // Initial centroids
		TreeMap<Integer, ArrayList<Integer>> best_cent_to_record = cent_to_record; // Initial mapping
		int[] best_labels = labels; // Store the labels which best minimize the cost function
		

		// Using Voronoi's algorithm, for each medoid m, identify point which
		// minimizes the cost in that cluster. Do this until convergence.
		for(iter = 0; iter < maxIter; iter++) {
			double[] medoid;
			double[] current_medoid;
			ArrayList<Integer> medoid_members;
			
			// Minimize cost in each cluster by selecting
			// best medoid for each cluster
			double systemCost = 0;
			for(int i = 0; i < k; i++) {
				medoid_members = cent_to_record.get(i);
				current_medoid = centroids.get(i);
				double current_cluster_cost = getCost(medoid_members, current_medoid);
				
				// For each point in the cluster, treat
				// every point as the medoid, calculate cost
				// and identify point which minimizes cluster cost
				for(Integer med : medoid_members) {
					medoid = data.getRow(med);
					double tmp_cost = getCost(medoid_members, medoid);
					if(tmp_cost < current_cluster_cost) {
						// Store medoid which minimized cost
						current_cluster_cost = tmp_cost;
						current_medoid = medoid;
					}
				}
				
				// Now set this medoid to the minimizing one
				centroids.set(i, current_medoid);
				systemCost += current_cluster_cost;
			}
			
			System.out.println("Last system cost: " + 
					oldCost + ". New system cost: " + 
					systemCost + ". Should converge == " + 
					(FastMath.abs(oldCost - systemCost) < minChange && iter != 0));
			
			// Now that best medoids have been selected, need 
			// to reassign system and ensure still decreasing...
			if(FastMath.abs(oldCost - systemCost) < minChange && iter!=0) { // Don't break on iter one...
				// Labels will be the same at this point.
				isTrained = true;
				converged = true;
				cost = systemCost;
				iter++; // Track iters used
				return;
			} else { // Has not yet converged
				if(systemCost < oldCost) { // Still decreasing...
					cost = systemCost;
					oldCost = cost;
					bestMedoids = centroids; // Initial centroids
					best_cent_to_record = cent_to_record; // Initial mapping
					best_labels = labels; // Store the labels which best minimize the cost function
				} else { // Is not decreasing, rather is increasing (if equal, would have exited)
					cost = oldCost;
					centroids = bestMedoids;
					cent_to_record = best_cent_to_record;
					labels = best_labels;
					converged = true;
					iter++;
					isTrained = true;
					return;
				}
				
				cent_to_record = assignClustersAndLabels();
			}
		}
		
		// If non-convergent
		isTrained = true;
	}
}
