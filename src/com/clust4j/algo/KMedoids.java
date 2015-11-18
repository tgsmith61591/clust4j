package com.clust4j.algo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.ClustUtils.SortedHashableIntSet;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class KMedoids extends AbstractKCentroidClusterer {
	final public static GeometricallySeparable DEF_DIST = Distance.MANHATTAN;
	
	/**
	 * Stores the indices of the current medoids. Each index,
	 * 0 thru k-1, corresponds to the class label for the cluster.
	 */
	private int[] medoid_indices = new int[k];
	
	/**
	 * Upper triangular, M x M matrix denoting distances between records.
	 * Is only populated during training phase and then set to null for 
	 * garbage collection, as a large-M matrix has a high space footprint: O(M choose 2).
	 * This is only needed during training and then can safely be collected
	 * to free up heap space.
	 */
	private double[][] dist_mat = null;
	
	
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
	 * The KMeans cluster assignment method leverages the super class' predict method,
	 * however KMedoids does <i>not</i>, as it internally uses the dist_mat for faster
	 * distance look-ups. This means more, less-generalized code, but faster execution time.
	 */
	@Override
	final protected TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabels() {
		/* Key is the closest centroid, value is the records that belong to it */
		TreeMap<Integer, ArrayList<Integer>> cent = new TreeMap<Integer, ArrayList<Integer>>();
		
		/* Loop over each record in the matrix */
		for(int rec = 0; rec < m; rec++) {
			double min_dist = Double.MAX_VALUE;
			int closest_cent = 0;
			
			/* Loop over every centroid, get calculate dist from record,
			 * identify the closest centroid to this record */
			for(int i = 0; i < k; i++) {
				final int centroid_idx = medoid_indices[i];
				final double dis = dist_mat[FastMath.min(rec, centroid_idx)][FastMath.max(rec, centroid_idx)];
				
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
	
	/**
	 * Returns the cost of a newly proposed system of medoids
	 * @param med_indices
	 * @return calculation of new system cost
	 */
	final double simulateSystemCost(final int[] med_indices, final double costToBeat) {
		/* Loop over each record in the matrix */
		double cst = 0;
		for(int rec = 0; rec < m; rec++) {
			double min_dist = Double.MAX_VALUE;
			
			/* Loop over every centroid, get calculate dist from record,
			 * identify the closest centroid to this record */
			for(int med_idx: med_indices) {
				final double dis = dist_mat[FastMath.min(rec, med_idx)][FastMath.max(rec, med_idx)];
				
				/* Track the current min distance. If dist
				 * is shorter than the previous min, assign
				 * new closest centroid to this record */
				if(dis < min_dist)
					min_dist = dis;
			}
			
			cst += min_dist;
			if(cst > costToBeat) // Hack to exit early if already higher than minCost...
				return costToBeat + 1;
		}
		
		return cst;
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
		
		if(verbose) info("beginning training segmentation for K = " + k);
		
		
		// Compute distance matrix, which is O(N^2) space, O(Nc2) time
		// We do this in KMedoids and not KMeans, because KMedoids uses
		// real points as medoids and not means for centroids, thus
		// the recomputation of distances is unnecessary with the dist mat
		dist_mat = ClustUtils.distanceMatrix(data, getDistanceMetric());
		
		
		if(verbose) info("calculated " + 
						dist_mat.length + " x " + 
						dist_mat.length + 
						" distance matrix");
				//+ ": HEAD = " + new MatrixFormatter()
				//	.format(new Array2DRowRealMatrix(dist_mat), 6));

		
		// Clusters initialized with randoms already in super
		// Initialize labels
		labels = new int[m];
		medoid_indices = init_centroid_indices;
		cent_to_record = assignClustersAndLabels();
		
		
		// State vars...
		// Once this config is no longer changing, global min reached
		double oldCost = getCostOfSystem();
		// Worst case will store up to M choose K...
		HashSet<SortedHashableIntSet> seen_medoid_combos = new HashSet<>();
		

		
		if(verbose)  {
			info("initial training system cost: " + oldCost );
		}
		
		
		for(iter = 0; iter < maxIter; iter++) {
		
			
			// Use the PAM (partitioning around medoids) algorithm
			// For each cluster in k...
			double min_cost = Double.MAX_VALUE;
			double new_cost = Double.MAX_VALUE;
			for(int i = 0; i < k; i++) {
				final int medoid_index = medoid_indices[i];
				final ArrayList<Integer> indices_in_cluster = cent_to_record.get(i);
				final double[] current_medoid = data.getRow(medoid_index);
				
				/*
				 * Need to take min here, because as the optimal medoids are found, from one
				 * cluster to another, cost_of_cluster will change (up or down). This ensures the
				 * min always being tracked.
				 */
				double cost_of_cluster = FastMath.min(getCost(indices_in_cluster, current_medoid), oldCost);
				
				if(verbose)
					info("optimizing medoid choice for cluster " + i + " (iter = " + (iter+1) + ")");
				
				
				// Track min for cluster
				int best_medoid_index = medoid_index;
				for(Integer o : indices_in_cluster) {
					if(o.intValue() == medoid_index) // Skip if it's the current medoid
						continue;
					
					// Create copy of medoids, set this med_idx to o
					final int[] copy_of_medoids = new int[k];
					System.arraycopy(medoid_indices, 0, copy_of_medoids, 0, k);
					copy_of_medoids[i] = o;
					
					// Create the sorted int set, see if these medoid combos have been seen before
					SortedHashableIntSet medoid_set = SortedHashableIntSet.fromArray(copy_of_medoids);
					if(seen_medoid_combos.contains(medoid_set))
						continue; // Micro hack!
					
					// Simulate cost, see if better...
					new_cost = simulateSystemCost(copy_of_medoids, oldCost);
					if(new_cost < cost_of_cluster) {
						cost_of_cluster = new_cost;
						if(verbose)
							info("new cost-minimizing system found; current cost: " + new_cost);
						
						best_medoid_index = o;
					}
					
					seen_medoid_combos.add(medoid_set); // Keep track of simulated medoid combos
				}
				
				// Have found optimal medoid to minimize cost in cluster...
				medoid_indices[i] = best_medoid_index;
				min_cost = cost_of_cluster;
			}
		
			// Check for stopping condition
			if( FastMath.abs(oldCost - min_cost) < minChange) { // convergence!
				// new_cost may sometimes retain Double.MAX_VALUE if never reassigned
				// in above loop, which means system hasn't changed and min is actually
				// the OLD cost
				oldCost = FastMath.min(oldCost, min_cost);
				
				if(verbose)
					info("training reached convergence at iteration "+ (iter+1) + 
							"; Total system cost: " + oldCost);
				
				converged = true;
				iter++;
				break;
			} else { // can get better... reassign clusters to new medoids, keep going.
				oldCost = min_cost;
				cent_to_record = assignClustersAndLabels();
			}
			
			
		} // End iter loop
		
		
		if(verbose && !converged) // KMedoids should always converge...
			warn("algorithm did not converge");
		
		
		cost = oldCost;
		isTrained = true;
		
		// Force GC to save space efficiency
		seen_medoid_combos = null;
		dist_mat = null;
	}
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEDOID;
	}
}
