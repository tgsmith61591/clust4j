package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Loggable;

public class BallTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -6424085914337479234L;

	public BallTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public BallTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	@Override
	public void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		// TODO
		
	}

	@Override
	public void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		// TODO
		
	}
}
