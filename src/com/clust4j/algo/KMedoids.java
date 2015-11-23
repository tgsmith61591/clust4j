package com.clust4j.algo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.ClustUtils.SortedHashableIntSet;
import com.clust4j.utils.Distance;

public class KMedoids extends AbstractKCentroidClusterer {
	
	/**
	 * Stores the indices of the current medoids. Each index,
	 * 0 thru k-1, corresponds to the class label for the cluster.
	 */
	private int[] medoid_indices = new int[k];
	
	/**
	 * Upper triangular, M x M matrix denoting distances between records.
	 * Is only populated during training phase and then set to null for 
	 * garbage collection, as a large-M matrix has a high space footprint: O(N^2).
	 * This is only needed during training and then can safely be collected
	 * to free up heap space.
	 */
	private double[][] dist_mat = null;
	
	private final GeometricallySeparable metric; // Just to save repeated function call overhead
	
	
	
	public KMedoids(final AbstractRealMatrix data, final int k) {
		this(data, new KMedoidsPlanner(k).setSep(Distance.MANHATTAN));
	}
	
	public KMedoids(final AbstractRealMatrix data, final KMedoidsPlanner builder) {
		super(data, builder);
		metric = getSeparabilityMetric();
	}
	
	
	
	public static class KMedoidsPlanner extends BaseKCentroidPlanner {
		public final static GeometricallySeparable DEF_DIST = Distance.MANHATTAN;
		public final static int DEF_MAX_ITER = 10; // Converges faster than KMeans, needs less
		
		public KMedoidsPlanner(int k) {
			super(k);
			super.setSep(DEF_DIST); // BY DEFAULT
			super.setMaxIter(DEF_MAX_ITER);
		}
		
		@Override
		public KMedoidsPlanner setSep(final GeometricallySeparable dist) {
			return (KMedoidsPlanner) super.setSep(dist);
		}
		
		public KMedoidsPlanner setMaxIter(final int max) {
			return (KMedoidsPlanner) super.setMaxIter(max);
		}

		public KMedoidsPlanner setMinChangeStoppingCriteria(final double min) {
			return (KMedoidsPlanner) super.setMinChangeStoppingCriteria(min);
		}
		
		@Override
		public KMedoidsPlanner setScale(final boolean scale) {
			return (KMedoidsPlanner) super.setScale(scale);
		}
		
		@Override
		public KMedoidsPlanner setVerbose(final boolean v) {
			return (KMedoidsPlanner) super.setVerbose(v);
		}
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
	 * Calculates the intracluster cost, only used in {@link #getCostOfSystem()}
	 * @param inCluster
	 * @return the sum of manhattan distances between vectors and the centroid
	 */
	@Override
	final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
		// Now calc the dissimilarity in the cluster
		double sumI = 0;
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			sumI += metric.getDistance(record, newCentroid);
		}
		
		return sumI; // The separability is never neg as returned by methods...
	}
	
	/**
	 * KMedoids-only hack to quickly get cost for cluster, 
	 * as the matrix is already cached
	 * @param indices
	 * @param med_idx
	 * @return
	 */
	protected double getCost(ArrayList<Integer> indices, final int med_idx) {
		double cost = 0;
		for(Integer idx: indices)
			cost += dist_mat[FastMath.min(idx, med_idx)][FastMath.max(idx, med_idx)];
		return cost;
	}
	
	
	@Override
	public String getName() {
		return "KMedoids";
	}

	@Override
	public void train() {
		synchronized(this) { // Synch because `isTrained` is a race condition
			if(isTrained)
				return;
			
			if(verbose) info("beginning training segmentation for K = " + k);
			
			final long start = System.currentTimeMillis();
			
			// Compute distance matrix, which is O(N^2) space, O(Nc2) time
			// We do this in KMedoids and not KMeans, because KMedoids uses
			// real points as medoids and not means for centroids, thus
			// the recomputation of distances is unnecessary with the dist mat
			dist_mat = ClustUtils.distanceMatrix(data, getSeparabilityMetric());
			
			/*System.out.println(new MatrixFormatter()
				.format(new Array2DRowRealMatrix(dist_mat, false)));*/
			
			
			if(verbose) info("calculated " + 
							dist_mat.length + " x " + 
							dist_mat.length + 
							" distance matrix in " + 
							LogTimeFormatter.millis( System.currentTimeMillis()-start , false));

			// Clusters initialized with randoms already in super
			// Initialize labels
			labels = new int[m];
			medoid_indices = init_centroid_indices;
			cent_to_record = assignClustersAndLabels();
			
			
			// State vars...
			// Once this config is no longer changing, global min reached
			double oldCost = getCostOfSystem(); // Tracks cost per iteration...
			
			// Worst case will store up to M choose K...
			HashSet<SortedHashableIntSet> seen_medoid_combos = new HashSet<>();
			

			
			if(verbose)  {
				info("initial training system cost: " + oldCost );
			}
			

			
			for(iter = 0; iter < maxIter; iter++) {
				// Use the PAM (partitioning around medoids) algorithm
				// For each cluster in k...
				// MUST BE DOUBLE MAX; if oldCost and no change, will
				// automatically "converge" and exit...
				double min_cost = oldCost; // The current minimum
			
				
				for(int i = 0; i < k; i++) {
					
					final int medoid_index = medoid_indices[i];
					final ArrayList<Integer> indices_in_cluster = cent_to_record.get(i);
					
					if(verbose)
						info("optimizing medoid choice for cluster " + 
							i + " (iter = " + (iter+1) + ") ");
					
					
					// Track min for cluster
					int best_medoid_index = medoid_index;
					for(Integer o : indices_in_cluster) {
						if(o.intValue() == medoid_index) // Skip if it's the current medoid
							continue;
						
						
						// Create copy of medoids, set this med_idx to o
						final int[] copy_of_medoids = VecUtils.copy(medoid_indices);
						copy_of_medoids[i] = o;
						
						
						// Create the sorted int set, see if these medoid combos have been seen before
						SortedHashableIntSet medoid_set = SortedHashableIntSet.fromArray(copy_of_medoids);
						if(seen_medoid_combos.contains(medoid_set))
							continue; // Micro hack!
						
						
						// Simulate cost, see if better...
						double simulated_cost = simulateSystemCost(copy_of_medoids, min_cost); // The simulated syst cost
						if(simulated_cost < min_cost) {
							min_cost = simulated_cost;
							if(verbose)
								trace("new cost-minimizing system found; current cost: " + simulated_cost );
							
							best_medoid_index = o;
						}
						
						seen_medoid_combos.add(medoid_set); // Keep track of simulated medoid combos
					}
					
					// Have found optimal medoid to minimize cost in cluster...
					medoid_indices[i] = best_medoid_index;
				}
			
				// Check for stopping condition
				if( FastMath.abs(oldCost - min_cost) < minChange) { // convergence!
					// new_cost may sometimes retain Double.MAX_VALUE if never reassigned
					// in above loop, which means system hasn't changed and min is actually
					// the OLD cost
					oldCost = FastMath.min(oldCost, min_cost);
					
					if(verbose) {
						info("training reached convergence at iteration "+ (iter+1) + 
								"; Total system cost: " + oldCost);
					
						info("model " + getKey() + " completed in " + 
								LogTimeFormatter.millis(System.currentTimeMillis()-start, false));
					}
					
					converged = true;
					iter++;
					break;
				} else { // can get better... reassign clusters to new medoids, keep going.
					if(verbose)
						info("convergence not yet reached, new min cost: " + min_cost);
					
					oldCost = min_cost;
					cent_to_record = assignClustersAndLabels();
				}
				
				
			} // End iter loop
			
			
			if(verbose && !converged) { // KMedoids should always converge...
				warn("algorithm did not converge");
				
				info("model " + getKey() + " completed in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-start, false));
			}
			
			
			cost = oldCost;
			isTrained = true;
			
			// Force GC to save space efficiency
			seen_medoid_combos = null;
			dist_mat = null;
			
		} // End synchronized
	} // End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEDOIDS;
	}
}
