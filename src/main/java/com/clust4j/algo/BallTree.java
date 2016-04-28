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

import java.util.HashSet;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.MinkowskiDistance;

/**
 * In computer science, a ball tree, balltree or metric tree, 
 * is a space partitioning data structure for organizing points 
 * in a multi-dimensional space. The ball tree gets its name from the 
 * fact that it partitions data points into a nested set of hyperspheres 
 * known as "balls". The resulting data structure has characteristics 
 * that make it useful for a number of applications, most notably 
 * nearest neighbor search.
 * @author Taylor G Smith
 * @see NearestNeighborHeapSearch
 * @see <a href="https://en.wikipedia.org/wiki/Ball_tree">Ball tree</a>
 */
public class BallTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -6424085914337479234L;
	public final static HashSet<Class<? extends GeometricallySeparable>> VALID_METRICS;
	static {
		VALID_METRICS = new HashSet<>();
		
		/*
		 * Want all distance metrics EXCEPT binary dist metrics
		 * and Canberra -- it tends to behave oddly on non-normalized data
		 */
		for(Distance dm: Distance.values()) {
			if(!dm.isBinaryDistance() && !dm.equals(Distance.CANBERRA)) {
				VALID_METRICS.add(dm.getClass());
			}
		}
		
		VALID_METRICS.add(MinkowskiDistance.class);
		VALID_METRICS.add(Distance.HAVERSINE.MI.getClass());
		VALID_METRICS.add(Distance.HAVERSINE.KM.getClass());
	}
	
	
	@Override protected boolean checkValidDistMet(GeometricallySeparable dist) {
		return VALID_METRICS.contains(dist.getClass());
	}
	
	
	
	public BallTree(final RealMatrix X) {
		super(X);
	}
	
	public BallTree(final RealMatrix X, int leaf_size) {
		super(X, leaf_size);
	}
	
	public BallTree(final RealMatrix X, DistanceMetric dist) {
		super(X, dist);
	}
	
	public BallTree(final RealMatrix X, Loggable logger) {
		super(X, logger);
	}
	
	public BallTree(final RealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public BallTree(final RealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	/**
	 * Constructor with logger and distance metric
	 * @param X
	 * @param dist
	 * @param logger
	 */
	public BallTree(final RealMatrix X, DistanceMetric dist, Loggable logger) {
		super(X, dist, logger);
	}
	
	protected BallTree(final double[][] X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	
	
	@Override
	void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		tree.node_bounds = new double[1][n_nodes][n_features];
	}

	@Override
	void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		int n_points = idx_end - idx_start, i, j, n_features = tree.N_FEATURES;
		double radius = 0;
		int[] idx_array = tree.idx_array;
		double[][] data = tree.data_arr;
		double[] centroid = tree.node_bounds[0][i_node], this_pt;
		
		// Determine centroid
		for(j = 0; j < n_features; j++)
			centroid[j] = 0;
		
		for(i = idx_start; i < idx_end; i++) {
			this_pt = data[idx_array[i]];
			
			for(j = 0; j < n_features; j++)
				centroid[j] += this_pt[j];
		}
		
		// Update centroids
		for(j = 0; j < n_features; j++) 
			centroid[j] /= n_points;
		
		
		// determine node radius
		for(i = idx_start; i < idx_end; i++)
			radius = FastMath.max(radius, 
					tree.rDist(centroid, data[idx_array[i]]));
		
		tree.node_data[i_node].radius = tree.dist_metric.partialDistanceToDistance(radius);
		tree.node_data[i_node].idx_start = idx_start;
		tree.node_data[i_node].idx_end = idx_end;
	}

	@Override
	final BallTree newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger) {
		return new BallTree(new Array2DRowRealMatrix(arr, false), leaf, dist, logger);
	}

	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double dist_pt = tree.dist(pt, tree.node_bounds[0][i_node]);
		return FastMath.max(0, dist_pt - tree.node_data[i_node].radius);
	}

	@Override
	double minRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.distanceToPartialDistance(minDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		return tree.dist_metric.distanceToPartialDistance(minDist(tree, i_node, pt));
	}

	/*
	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double dist_pt = tree.dist(pt, tree.node_bounds[0][i_node]);
		return dist_pt + tree.node_data[i_node].radius;
	}

	@Override
	double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		return tree.dist_metric.distanceToPartialDistance(maxDist(tree, i_node, pt));
	}
	*/

	@Override
	double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.distanceToPartialDistance(maxDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	double maxDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		double dist_pt = tree1.dist(tree2.node_bounds[0][iNode2], tree1.node_bounds[0][iNode1]);
		return dist_pt + tree1.node_data[iNode1].radius + tree2.node_data[iNode2].radius;
	}

	@Override
	double minDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		double dist_pt = tree1.dist(tree2.node_bounds[0][iNode2], 
									tree1.node_bounds[0][iNode1]);
		return FastMath.max(0, 
				(dist_pt 
				- tree1.node_data[iNode1].radius
				- tree2.node_data[iNode2].radius));
	}

	@Override
	void minMaxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt, MutableDouble minDist, MutableDouble maxDist) {
		double dist_pt = tree.dist(pt, tree.node_bounds[0][i_node]);
		double rad = tree.node_data[i_node].radius;
		minDist.value = FastMath.max(0, dist_pt - rad);
		maxDist.value = dist_pt + rad;
	}
}
