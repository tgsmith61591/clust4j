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
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.IllegalClusterStateException;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.Pairwise;
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
 * solution. For this reason, clust4j's implementation of KMedoids almost always surpasses
 * the performance of {@link KMeans}, however it can typically take longer  as well.
 * 
 * @see {@link AbstractPartitionalClusterer}
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
final public class KMedoids extends AbstractCentroidClusterer {
	
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
	volatile private double[] weighted_vector = null;
	
	/**
	 * Map the index to the WSS
	 */
	volatile private TreeMap<Integer, Double> med_to_wss = new TreeMap<>();

	protected KMedoids(final RealMatrix data, final RealVector weights) {
		this(data, weights, DEF_K);
	}
	
	protected KMedoids(final RealMatrix data) {
		this(data, DEF_K);
	}

	protected KMedoids(final RealMatrix data, final RealVector weights, final int k) {
		this(data, weights, new KMedoidsParameters(k).setMetric(Distance.MANHATTAN));
	}

	protected KMedoids(final RealMatrix data, final int k) {
		this(data, new KMedoidsParameters(k).setMetric(Distance.MANHATTAN));
	}

	protected KMedoids(final RealMatrix data, final RealVector weights, final KMedoidsParameters planner) {
		super(data, planner);
		weighted_vector = weights.toArray();
		// Check if is Manhattan
		if(!this.dist_metric.equals(Distance.MANHATTAN)) {
			warn("KMedoids is intented to run with Manhattan distance, WSS/BSS computations will be inaccurate");
			//this.dist_metric = Distance.MANHATTAN; // idk that we want to enforce this...
		}
	}

	protected KMedoids(final RealMatrix data, final KMedoidsParameters planner) {
		this(data, new ArrayRealVector(data.getRowDimension(), 1), planner);
	}


	
	
	@Override
	public String getName() {
		return "KMedoids";
	}
	
	@Override
	protected KMedoids fit() {
		synchronized(fitLock) {	
		
			if(null != labels) // already fit
				return this;
			
			final LogTimer timer = new LogTimer();
			final double[][] X = data.getData();
			final double nan = Double.NaN;
			
			
			// Corner case: K = 1 or all singular
			if(1 == k) {
				labelFromSingularK(X);
				fitSummary.add(new Object[]{ iter, converged, 
					tss, // tss
					tss, // avg per cluster
					tss, // wss
					nan, // bss (none)
					timer.wallTime() });
				sayBye(timer);
				return this;
			}
			
			
			// We do this in KMedoids and not KMeans, because KMedoids uses
			// real points as medoids and not means for centroids, thus
			// the recomputation of distances is unnecessary with the dist mat
			dist_mat = Pairwise.getDistance(X, getSeparabilityMetric(), true, false);
			info("distance matrix computed in " + timer.toString());
			
			// Initialize labels
			medoid_indices = init_centroid_indices;
			
			
			ClusterAssignments clusterAssignments;
			MedoidReassignmentHandler rassn;
			int[] newMedoids = medoid_indices;
			
			// Cost vars
			double bestCost = Double.POSITIVE_INFINITY, 
				   maxCost = Double.NEGATIVE_INFINITY,
				   avgCost = Double.NaN, wss_sum = nan;
			
			
			// Iterate while the cost decreases:
			boolean convergedFromCost = false; // from cost or system changes?
			boolean configurationChanged = true;
			while( configurationChanged
				&& iter < maxIter ) {
				
				/*
				 * 1. In each cluster, make the point that minimizes 
				 *    the sum of distances within the cluster the medoid
				 */
				try {
					clusterAssignments = assignClosestMedoid(newMedoids);
				} catch(IllegalClusterStateException ouch) {
					exitOnBadDistanceMetric(X, timer);
					return this;
				}
				
				
				/*
				 * 1.5 The entries are not 100% equal, so we can (re)assign medoids...
				 */
				try {
					rassn = new MedoidReassignmentHandler(clusterAssignments);
				} catch(IllegalClusterStateException ouch) {
					exitOnBadDistanceMetric(X, timer);
					return this;
				}
				
				/*
				 * 1.75 This happens in the case of bad kernels that cause
				 * infinities to propagate... we can't segment the input
				 * space and need to just return a single cluster.
				 */
				if(rassn.new_clusters.size() == 1) {
					this.k = 1;
					warn("(dis)similarity metric cannot partition space without propagating Infs. Returning one cluster");
					
					labelFromSingularK(X);
					fitSummary.add(new Object[]{ iter, converged, 
							tss, // tss
							tss, // avg per cluster
							tss, // wss
							nan, // bss (none) 
							timer.wallTime() });
					sayBye(timer);
					return this;
				}

				
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
				 * 3. Update the costs
				 */
				converged = lastIteration || (convergedFromCost = FastMath.abs(wss_sum - bestCost) < tolerance);
				double tmp_wss_sum = rassn.new_clusters.total_cst;
				double tmp_bss = tss - tmp_wss_sum;

				// Check whether greater than max
				if(tmp_wss_sum > maxCost)
					maxCost = tmp_wss_sum;

				if(tmp_wss_sum < bestCost) {
					bestCost = wss_sum = tmp_wss_sum;
					labels = rassn.new_clusters.assn; // will be medoid idcs until encoded at end
					med_to_wss = rassn.new_clusters.costs;
					centroids = rassn.centers;
					medoid_indices = newMedoids;
					bss = tmp_bss;
					
					// get avg cost
					avgCost = wss_sum / (double)k;
				}

				if(converged) {
					reorderLabelsAndCentroids();
				}
				
				/*
				 * 3.5 If this is the last one, it'll show the wss and bss
				 */
				fitSummary.add(new Object[]{ iter, 
					converged,
					tss, 
					avgCost, 
					wss_sum, 
					bss, 
					timer.wallTime()
				});
				

				iter++;
				configurationChanged = !converged;
			}
			
			if(!converged)
				warn("algorithm did not converge");
			else 
				info("algorithm converged due to " + 
				(convergedFromCost ? "cost minimization" : "harmonious state"));
			
				
			// wrap things up, create summary..
			sayBye(timer);
			
			return this;
		}
		
	} // End train
	
	
	/**
	 * Some metrics produce entirely equal dist matrices...
	 */
	private void exitOnBadDistanceMetric(double[][] X, LogTimer timer) {
		warn("distance metric (" + dist_metric + ") produced entirely equal distances");
		labelFromSingularK(X);
		fitSummary.add(new Object[]{ iter, converged, tss, tss, tss, Double.NaN, Double.NaN, timer.wallTime() });
		sayBye(timer);
	}
	
	
	private ClusterAssignments assignClosestMedoid(int[] medoidIdcs) {
		double minDist;
		boolean all_tied = true;
		int nearest, rowIdx, colIdx;
		final int[] assn = new int[m];
		final double[] costs = new double[m];
		for(int i = 0; i < m; i++) {
			boolean is_a_medoid = false;
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
					is_a_medoid = true;
					break;
				}
				
				rowIdx = FastMath.min(i, medoid);
				colIdx = FastMath.max(i, medoid);
				
				if((dist_mat[rowIdx][colIdx] * weighted_vector[i])< minDist) {
					minDist = dist_mat[rowIdx][colIdx] * weighted_vector[i];
					nearest = medoid;
				}
			}
			
			/*
			 * If all of the distances are equal, we can end up with a -1 idx...
			 */
			if(-1 == nearest)
				nearest = medoidIdcs[getSeed().nextInt(k)]; // select random nearby
			if(!is_a_medoid)
				all_tied = false;
			
			
			assn[i]	 = nearest;
			costs[i] = minDist; 
		}
		
		
		/*
		 * If everything is tied, we need to bail. Shouldn't happen, now
		 * that we explicitly check earlier on... but we can just label from
		 * a singular K at this point.
		 */
		if(all_tied) {
			throw new IllegalClusterStateException("entirely "
				+ "stochastic process: all distances are equal");
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
				int rowIdx, colIdx, bestMedoid = 0; // start at 0, not -1 in case of all ties...
				for(int a: members) { // check cost if A is the medoid...
					
					medoidCost = 0.0;
					for(int b: members) {
						if(a == b)
							continue;
						
						rowIdx = FastMath.min(a, b);
						colIdx = FastMath.max(a, b);
						
						medoidCost += dist_mat[rowIdx][colIdx] * weighted_vector[b];
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
		return Algo.KMEDOIDS;
	}
	
	@Override
	protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","TSS","Avg Clust. Cost","Min WSS","Max BSS","Wall"
		};
	}
	
	/**
	 * Reorder the labels in order of appearance using the 
	 * {@link LabelEncoder}. Also reorder the centroids to correspond
	 * with new label order
	 */
	protected void reorderLabelsAndCentroids() {
		
		/*
		 *  reorder labels...
		 */
		final LabelEncoder encoder = new LabelEncoder(labels).fit();
		labels = encoder.getEncodedLabels();
		
		int i = 0;
		centroids = new ArrayList<>();
		int[] classes = encoder.getClasses();
		for(int claz: classes) {
			centroids.add(data.getRow(claz)); // an index, not a counter 0 thru k
			wss[i++] = med_to_wss.get(claz);
		}
	}
	
	@Override final protected GeometricallySeparable defMetric() { return KMedoids.DEF_DIST; }
}
