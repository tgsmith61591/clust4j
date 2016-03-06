package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.VecUtils;

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
 * <p>
 * clust4j utilizes the <a href="https://en.wikipedia.org/wiki/Lloyd%27s_algorithm">
 * Voronoi iteration</a> technique to identify clusters. Alternative greedy searches, 
 * including PAM (partitioning around medoids), are faster yet may not find the optimal
 * solution.
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
	
	
	
	public KMedoids(final AbstractRealMatrix data) {
		this(data, DEF_K);
	}
	
	public KMedoids(final AbstractRealMatrix data, final int k) {
		this(data, new KMedoidsPlanner(k).setSep(Distance.MANHATTAN));
	}
	
	public KMedoids(final AbstractRealMatrix data, final KMedoidsPlanner planner) {
		super(data, planner);
	}
	
	
	
	public static class KMedoidsPlanner extends CentroidClustererPlanner {
		private static final long serialVersionUID = -3288579217568576647L;
		
		private InitializationStrategy strat = DEF_INIT;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_CONVERGENCE_TOLERANCE;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private int k = DEF_K;
		
		public KMedoidsPlanner() { }
		public KMedoidsPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public KMedoids buildNewModelInstance(final AbstractRealMatrix data) {
			return new KMedoids(data, this.copy());
		}
		
		@Override
		public KMedoidsPlanner copy() {
			return new KMedoidsPlanner(k)
				.setMaxIter(maxIter)
				.setConvergenceCriteria(minChange)
				.setScale(scale)
				.setSep(dist)
				.setVerbose(verbose)
				.setSeed(seed)
				.setNormalizer(norm)
				.setInitializationStrategy(strat);
		}
		
		@Override
		public InitializationStrategy getInitializationStrategy() {
			return strat;
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
		public double getConvergenceTolerance() {
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

		@Override
		public KMedoidsPlanner setConvergenceCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public KMedoidsPlanner setInitializationStrategy(InitializationStrategy init) {
			this.strat = init;
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
	
	
	
	@Override
	public String getName() {
		return "KMedoids";
	}
	
	@Override
	public KMedoids fit() {
		synchronized(this) {
			
			try {
				if(null != labels) // already fit
					return this;
				
				final LogTimer timer = new LogTimer();
				final double[][] X = data.getData();
				
				// Corner case: K = 1
				if(1 == k) {
					labelFromSingularK(X);
					fitSummary.add(new Object[]{ iter, converged, cost, cost, cost, timer.wallTime() });
					sayBye(timer);
					return this;
				}
				
				
				// We do this in KMedoids and not KMeans, because KMedoids uses
				// real points as medoids and not means for centroids, thus
				// the recomputation of distances is unnecessary with the dist mat
				dist_mat = ClustUtils.distanceUpperTriangMatrix(X, getSeparabilityMetric());
				info("distance matrix computed in " + timer.toString());
				
				
				// Initialize labels
				medoid_indices = init_centroid_indices;
				
				
				ClusterAssignments clusterAssignments;
				MedoidReassignmentHandler rassn;
				int[] newMedoids = medoid_indices;
				
				// Cost vars
				cost = Double.POSITIVE_INFINITY;
				double bestCost = Double.POSITIVE_INFINITY, 
					   maxCost = Double.NEGATIVE_INFINITY,
					   avgCost = Double.NaN;
				
				
				// Iterate while the cost decreases:
				boolean convergedFromCost = false; // from cost or system changes?
				boolean configurationChanged = true;
				while( configurationChanged
					&& iter < maxIter ) {
					
					/*
					 * 1. In each cluster, make the point that minimizes 
					 *    the sum of distances within the cluster the medoid
					 */
					clusterAssignments = assignClosestMedoid(newMedoids);
					rassn = new MedoidReassignmentHandler(clusterAssignments);

					
					/*
					 * 2. Reassign each point to the cluster defined by the 
					 *    closest medoid determined in the previous step.
					 */
					newMedoids = rassn.reassignedMedoidIdcs;

					
					/*
					 * 2.5 Determine whether configuration changed
					 */
					boolean lastIteration = VecUtils.equalsExactly(newMedoids, medoid_indices);
					
					
					/*
					 * 3. Update the fit summary item
					 */
					fitSummary.add(new Object[]{ iter, 
						converged = lastIteration 
							|| (convergedFromCost = FastMath.abs(cost - bestCost) < tolerance), 
						maxCost, cost, avgCost, timer.wallTime() });
					
					/*
					 * 4. Update the costs
					 */
					double tmpCost = rassn.new_clusters.total_cst;
					avgCost = tmpCost / (double)k;
					if(tmpCost > maxCost)
						maxCost = tmpCost;
					
					if(tmpCost < bestCost) {
						bestCost = tmpCost;
						cost = bestCost;
						labels = rassn.new_clusters.assn; // will be medoid idcs until encoded at end
						centroids = rassn.centers;
						medoid_indices = newMedoids;
					}
					

					iter++;
					configurationChanged = !converged;
				}
				
				if(!converged)
					warn("algorithm did not converge");
				else 
					info("algorithm converged due to " + 
					(convergedFromCost ? "cost minimization" : "harmonious state"));
				
					
				
				// wrap things up, create summary..
				reorderLabelsAndCentroids();
				sayBye(timer);
				
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synchronized
	} // End train
	
	
	private ClusterAssignments assignClosestMedoid(int[] medoidIdcs) {
		double minDist;
		int nearest, rowIdx, colIdx;
		final int[] assn = new int[m];
		final double[] costs = new double[m];
		for(int i = 0; i < m; i++) {
			minDist = Double.POSITIVE_INFINITY;
			
			/*
			 * The dist_mat is already computed. We just need to traverse
			 * the upper triangular matrix and identify which corresponding
			 * minimum distance per record.
			 */
			nearest = -1;
			for(int medoid: medoidIdcs) {
				
				// Corner case: i is a medoid
				if(i == medoid) {
					nearest = medoid;
					minDist = dist_mat[i][i];
					break;
				}
				
				rowIdx = FastMath.min(i, medoid);
				colIdx = FastMath.max(i, medoid);
				
				if(dist_mat[rowIdx][colIdx] < minDist) {
					minDist = dist_mat[rowIdx][colIdx];
					nearest = medoid;
				}
			}
			
			assn[i]	 = nearest;
			costs[i] = minDist; 
		}
		
		return new ClusterAssignments(assn, costs);
	}
	
	
	/**
	 * Handles medoids reassignments and cost minimizations.
	 * In the Voronoi iteration algorithm, after we've identified the new
	 * cluster assignment, for each cluster, we select the medoid which minimized
	 * intra-cluster variance. Theoretically, this could result in a re-org of clusters,
	 * so we use the new medoid indices to create a new {@link ClusterAssignments} object
	 * as the last step. If the cost does not change in the last step, we know we've
	 * reached convergence.
	 * @author Taylor G Smith
	 */
	private class MedoidReassignmentHandler {
		final ClusterAssignments init_clusters;
		final ArrayList<double[]> centers = new ArrayList<double[]>(k);
		final int[] reassignedMedoidIdcs  = new int[k];
		
		// Holds the costs of each cluster in order
		final ClusterAssignments new_clusters;
		
		/**
		 * Def constructor
		 * @param assn - new medoid assignments
		 */
		MedoidReassignmentHandler(ClusterAssignments assn) {
			this.init_clusters = assn;
			medoidAssn();
			this.new_clusters = assignClosestMedoid(reassignedMedoidIdcs);
		}
		
		void medoidAssn() {
			ArrayList<Integer> members;
			
			int i = 0;
			for(Map.Entry<Integer, ArrayList<Integer>> pair: init_clusters.entrySet()) {
				members = pair.getValue();
				
				double medoidCost, minCost = Double.POSITIVE_INFINITY;
				int rowIdx, colIdx, bestMedoid = -1;
				for(int a: members) { // check cost if A is the medoid...
					
					medoidCost = 0.0;
					for(int b: members) {
						if(a == b)
							continue;
						
						rowIdx = FastMath.min(a, b);
						colIdx = FastMath.max(a, b);
						
						medoidCost += dist_mat[rowIdx][colIdx];
					}
					
					if(medoidCost < minCost) {
						minCost = medoidCost;
						bestMedoid = a;
					}
				}
				
				this.reassignedMedoidIdcs[i] = bestMedoid;
				this.centers.add(data.getRow(bestMedoid));
				i++;
			}
		}
	}
	
	/**
	 * Simple container for handling cluster assignments. Given
	 * an array of length m of medoid assignments, and an array of length m
	 * of distances to the medoid, organize the new clusters and compute the total
	 * cost of the new system.
	 * @author Taylor G Smith
	 */
	private class ClusterAssignments extends TreeMap<Integer, ArrayList<Integer>> {
		private static final long serialVersionUID = -7488380079772496168L;
		final int[] assn;
		TreeMap<Integer, Double> costs; // maps medoid idx to cluster cost
		double total_cst;
		
		ClusterAssignments(int[] assn, double[] costs) {
			super();
			
			// should be equal in length to costs arg
			this.assn = assn;
			this.costs = new TreeMap<>();
			
			int medoid;
			double cost;
			ArrayList<Integer> ref;
			for(int i = 0; i < assn.length; i++) {
				medoid = assn[i];
				cost = costs[i];
				
				ref = get(medoid); // helps avoid double lookup later
				if(null == ref) { // not here.
					ref =  new ArrayList<Integer>();
					ref.add(i);
					put(medoid, ref);
					this.costs.put(medoid, cost);
				} else {
					ref.add(i);
					double d = this.costs.get(medoid);
					this.costs.put(medoid, d + cost);
				}
				
				total_cst += cost;
			}
		}
	}

	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEDOIDS;
	}
	
	@Override
	protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","Max Cost","Min Cost","Avg Clust. Cost","Wall"
		};
	}
}
