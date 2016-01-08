package com.clust4j.algo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.ClustUtils.SortedHashableIntSet;
import com.clust4j.utils.Distance;

/**
 * <a href="https://en.wikipedia.org/wiki/K-medoids">KMedoids</a> is
 * a clustering algorithm related to the {@link KMeans} algorithm and the 
 * medoidshift algorithm. Both the KMeans and KMedoids algorithms are 
 * partitional (breaking the dataset up into groups) and both attempt 
 * to minimize the distance between points labeled to be in a cluster 
 * and a point designated as the center of that cluster. In contrast to 
 * the KMeans algorithm, KMedoids chooses datapoints as centers (medoids 
 * or exemplars) and works with an arbitrary matrix of distances between 
 * datapoints instead of Euclidean distance (l2 norm). This method was proposed in 
 * 1987 for the work with Manhattan distance (l1 norm) and other distances.
 * 
 * @see {@link AbstractPartitionalClusterer}
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public class KMedoids extends AbstractCentroidClusterer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -4468316488158880820L;
	final public static GeometricallySeparable DEF_DIST = Distance.MANHATTAN;
	final public static int DEF_MAX_ITER = 10;

	/**
	 * Stores the indices of the current medoids. Each index,
	 * 0 thru k-1, corresponds to the class label for the cluster.
	 */
	volatile private int[] medoid_indices = new int[k];
	
	/**
	 * Upper triangular, M x M matrix denoting distances between records.
	 * Is only populated during training phase and then set to null for 
	 * garbage collection, as a large-M matrix has a high space footprint: O(N^2).
	 * This is only needed during training and then can safely be collected
	 * to free up heap space.
	 */
	volatile private double[][] dist_mat = null;
	
	
	
	
	public KMedoids(final AbstractRealMatrix data, final int k) {
		this(data, new KMedoidsPlanner(k).setSep(Distance.MANHATTAN));
	}
	
	public KMedoids(final AbstractRealMatrix data, final KMedoidsPlanner planner) {
		super(data, planner);
	}
	
	
	
	public static class KMedoidsPlanner extends CentroidClustererPlanner {
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_MIN_CHNG;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private int k;
		
		public KMedoidsPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public KMedoids buildNewModelInstance(final AbstractRealMatrix data) {
			return new KMedoids(data, this);
		}
		
		@Override
		public KMedoidsPlanner copy() {
			return new KMedoidsPlanner(k)
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
		public KMedoidsPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public KMedoidsPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}

		public KMedoidsPlanner setMinChangeStoppingCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public KMedoidsPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public KMedoidsPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public KMedoidsPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public KMedoidsPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	
	
	

	/**
	 * The KMeans cluster assignment method leverages the super class' predict method,
	 * however KMedoids does <i>not</i>, as it internally uses the dist_mat for faster
	 * distance look-ups. This means more, less-generalized code, but faster execution time.
	 */
	final TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabelsInPlace() {
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
	private final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
		// Now calc the dissimilarity in the cluster
		double sumI = 0;
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			sumI += getSeparabilityMetric().getDistance(record, newCentroid);
		}
		
		return sumI; // The separability is never neg as returned by methods...
	}
	
	@Override
	public String getName() {
		return "KMedoids";
	}

	@Override
	public KMedoids fit() {
		synchronized(this) { // Synch because alters internal structs
			
			try {
				if(null!=labels) // Already have fit this model
					return this;
				
				info("beginning training segmentation for K = " + k);
				final long start = System.currentTimeMillis();
				
				// Compute distance matrix, which is O(N^2) space, O(Nc2) time
				// We do this in KMedoids and not KMeans, because KMedoids uses
				// real points as medoids and not means for centroids, thus
				// the recomputation of distances is unnecessary with the dist mat
				dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
				
				/*System.out.println(new MatrixFormatter()
					.format(new Array2DRowRealMatrix(dist_mat, false)));*/
				
				
				info("calculated " + 
					m + " x " + m + 
					" distance matrix in " + 
					LogTimeFormatter.millis( System.currentTimeMillis()-start , false));
	
				// Clusters initialized with randoms already in super
				// Initialize labels
				labels = new int[m];
				medoid_indices = init_centroid_indices;
				cent_to_record = assignClustersAndLabelsInPlace();
				
				
				// State vars...
				// Once this config is no longer changing, global min reached
				double oldCost = getCostOfSystem(); // Tracks cost per iteration...
				
				// Worst case will store up to M choose K...
				HashSet<SortedHashableIntSet> seen_medoid_combos = new HashSet<>();
				info("initial training system cost: " + oldCost );
				
	
				long iterStart = System.currentTimeMillis();
				for(iter = 0; iter < maxIter; iter++) {
					// Use the PAM (partitioning around medoids) algorithm
					// For each cluster in k...
					// MUST BE DOUBLE MAX; if oldCost and no change, will
					// automatically "converge" and exit...
					double min_cost = oldCost; // The current minimum
				
					
					for(int i = 0; i < k; i++) {
						
						final int medoid_index = medoid_indices[i];
						final ArrayList<Integer> indices_in_cluster = cent_to_record.get(i);
						
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
						
						info("training reached convergence at iteration "+ (iter+1) + " (avg iteration time: " + 
							LogTimeFormatter.millis( (long) ((long)(System.currentTimeMillis()-iterStart)/(double)(iter+1)), false) + ")");
						info("Total system cost: " + oldCost);
						
						converged = true;
						iter++;
						break;
					} else { // can get better... reassign clusters to new medoids, keep going.
						info("algorithm has not converged yet; new min cost: " + min_cost);
						
						oldCost = min_cost;
						cent_to_record = assignClustersAndLabelsInPlace();
					}
					
					
				} // End iter loop
				
				
					if(!converged) // KMedoids should always converge...
						warn("algorithm did not converge");
					
					info("model " + getKey() + " completed in " + 
							LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
							System.lineSeparator());
				
				
				cost = oldCost;
				
				// Force GC to save space efficiency
				seen_medoid_combos = null;
				dist_mat = null;
				
				
				reorderLabels();
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synchronized
	} // End train
	
	private double getCostOfSystem() {
		double cost = 0;
		double[] oid;
		ArrayList<Integer> medoid_members;
		for(Map.Entry<Integer, ArrayList<Integer>> medoid_entry : cent_to_record.entrySet()) {
			oid = centroids.get(medoid_entry.getKey()); //cent-med-oid
			medoid_members = medoid_entry.getValue();
			cost += getCost(medoid_members, oid);
		}
		
		return cost;
	}
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEDOIDS;
	}
	
	final private void reorderLabels() {
		// Assign medoid indices records to centroids
		centroids = new ArrayList<>();
		
		
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
				newCentroids.put(idx, data.getRow(medoid_indices[labels[i]]) );
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
