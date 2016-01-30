package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;

public class BallTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -6424085914337479234L;

	
	
	public BallTree(final AbstractRealMatrix X) {
		super(X, DEF_LEAF_SIZE, DEF_DIST);
	}
	
	public BallTree(final AbstractRealMatrix X, int leaf_size) {
		super(X, leaf_size, DEF_DIST);
	}
	
	public BallTree(final AbstractRealMatrix X, DistanceMetric dist) {
		super(X, DEF_LEAF_SIZE, dist);
	}
	
	public BallTree(final AbstractRealMatrix X, Loggable logger) {
		super(X, DEF_LEAF_SIZE, DEF_DIST, logger);
	}
	
	public BallTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public BallTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	
	
	@Override
	void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		tree.node_bounds = new double[1][n_nodes][n_features];
	}

	@Override
	void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		int n_points = idx_end - idx_start, i, j;
		double radius = 0;
		int[] idx_array = tree.idx_array;
		double[][] data = tree.data_arr;
		double[] centroid = tree.node_bounds[0][i_node], this_pt;
		
		// Determine centroid
		for(j = 0; j < N_FEATURES; j++)
			centroid[j] = 0;
		
		boolean lastIter = false;
		for(i = idx_start; i < idx_end; i++) {
			lastIter = i == idx_end - 1;
			this_pt = data[idx_array[i]];
			
			for(j = 0; j < N_FEATURES; j++) {
				centroid[j] += this_pt[j];
				
				if(lastIter) // Added in to save one O(N) pass
					centroid[j] /= n_points;
			}
		}
		
		// Original code included this AFTER previous loop,
		// but this can be rolled in to final iter of the loop
		// for optimization.
		// for(j = 0; j < N_FEATURES; j++) centroid[j] /= n_points;
		
		
		// determine node radius
		for(i = idx_start; i < idx_end; i++)
			radius = FastMath.max(radius, 
					tree.rDist(centroid, data[idx_array[i]]));
		
		tree.node_data[i_node].radius = tree.dist_metric.partialDistanceToDistance(radius);
		tree.node_data[i_node].idx_start = idx_start;
		tree.node_data[i_node].idx_end = idx_end;
	}

	@Override
	BallTree newInstance(double[][] arr, int leaf, DistanceMetric dist) {
		return newInstance(arr, leaf, dist, null);
	}

	@Override
	BallTree newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger) {
		return new BallTree(new Array2DRowRealMatrix(arr, false), leaf, dist, logger);
	}

	@Override
	double minRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.distanceToPartialDistance(minDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		return tree.dist_metric.distanceToPartialDistance(minDist(tree, i_node, pt));
	}

	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double dist_pt = tree.dist(pt, tree.node_bounds[0][i_node]);
		return FastMath.max(0, dist_pt - tree.node_data[i_node].radius);
	}

	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double dist_pt = tree.dist(pt, tree.node_bounds[0][i_node]);
		return dist_pt + tree.node_data[i_node].radius;
	}

	@Override
	double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		return tree.dist_metric.distanceToPartialDistance(maxDist(tree, i_node, pt));
	}

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
		double dist_pt = tree1.dist(tree2.node_bounds[0][iNode2], tree1.node_bounds[0][iNode1]);
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
