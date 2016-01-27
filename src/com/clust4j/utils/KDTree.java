package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Loggable;

public class KDTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -3744545394278454548L;
	

	public KDTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public KDTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	@Override
	public void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		tree.node_bounds = new double[2][n_nodes][n_features];
	}

	@Override
	public void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		// TODO
		
	}
	
	// TODO
}
