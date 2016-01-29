package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

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
		int n = tree.data_arr[0].length, n_points = idx_end - idx_start, i, j;
		double radius;
		int[] idx_array = tree.idx_array;
		double[][] data = tree.data_arr;
		double[][][] centroid = tree.node_bounds;
		
		// TODO
		
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
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] a) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] a) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double maxDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double minDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double minMaxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt, double lb, double ub) {
		// TODO Auto-generated method stub
		return 0;
	}
}
