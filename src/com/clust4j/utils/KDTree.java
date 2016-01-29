package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;

public class KDTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -3744545394278454548L;
	
	
	
	public KDTree(final AbstractRealMatrix X) {
		super(X, DEF_LEAF_SIZE, DEF_DIST);
	}
	
	public KDTree(final AbstractRealMatrix X, int leaf_size) {
		super(X, leaf_size, DEF_DIST);
	}
	
	public KDTree(final AbstractRealMatrix X, DistanceMetric dist) {
		super(X, DEF_LEAF_SIZE, dist);
	}
	
	public KDTree(final AbstractRealMatrix X, Loggable logger) {
		super(X, DEF_LEAF_SIZE, DEF_DIST, logger);
	}

	public KDTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public KDTree(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	
	
	@Override
	void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		tree.node_bounds = new double[2][n_nodes][n_features];
	}

	@Override
	void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		int n_features = tree.data_arr[0].length, i, j;
		double rad = 0;
		
		double[] lowerBounds = tree.node_bounds[0][i_node];
		double[] upperBounds = tree.node_bounds[1][i_node];
		double[][] data = tree.data_arr;
		int[] idx_array = tree.idx_array;
		double[] data_row;
		
		// Get node bounds
		for(j = 0; j < n_features; j++) {
			lowerBounds[j] = Double.POSITIVE_INFINITY;
			upperBounds[j] = Double.NEGATIVE_INFINITY;
		}
		
		// Compute data range
		for(i = idx_start; i < idx_end; i++) {
			data_row = data[idx_array[i]];
			
			for(j = 0; j < n_features; j++) {
				lowerBounds[j] = FastMath.min(lowerBounds[j], data_row[j]);
				upperBounds[j] = FastMath.max(upperBounds[j], data_row[j]);
			}
			
			// The python code does not increment up to the range boundary,
			// the java for loop does. So we must decrement j by one.
			j--;
			
			if( Double.isInfinite(tree.dist_metric.getP()) )
				rad = FastMath.max(rad, 0.5 * (upperBounds[j] - lowerBounds[j]));
			else
				rad += FastMath.pow(0.5 * FastMath.abs(upperBounds[j] - lowerBounds[j]), 
						tree.dist_metric.getP());
		}
		
		tree.node_data[i_node].idx_start = idx_start;
		tree.node_data[i_node].idx_end = idx_end;
		tree.node_data[i_node].radius = Math.pow(rad, 1d / tree.dist_metric.getP());
	}
	
	@Override
	KDTree newInstance(double[][] arr, int leaf, DistanceMetric dist) {
		return newInstance(arr, leaf, dist, null);
	}

	@Override
	KDTree newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger) {
		return new KDTree(new Array2DRowRealMatrix(arr, false), leaf, dist, logger);
	}

	@Override
	double minRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = minRDist(tree, i_node, pt);
		if(Double.isInfinite(tree.dist_metric.getP()))
			return d;
		return Math.pow(d, 1d / tree.dist_metric.getP());
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] a) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = maxRDist(tree, i_node, pt);
		if(Double.isInfinite(tree.dist_metric.getP()))
			return d;
		return Math.pow(d, 1d / tree.dist_metric.getP());
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
