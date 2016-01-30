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
		double d, d1, d2, rdist = 0.0;
		int j;
		
		for(j = 0; j < N_FEATURES; j++) {
			d1 = tree1.node_bounds[0][iNode1][j] - tree2.node_bounds[1][iNode2][j];
			d2 = tree2.node_bounds[0][iNode2][j] - tree1.node_bounds[1][iNode1][j];
			d = (d1 + FastMath.abs(d1)) + (d2 + FastMath.abs(d2));
			
			if(tree1.dist_metric.getP() == Double.POSITIVE_INFINITY)
				rdist = FastMath.max(rdist, 0.5 * d);
			else
				rdist += FastMath.pow(0.5 * d, tree1.dist_metric.getP());
		}
		
		return rdist;
	}
	
	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = minRDist(tree, i_node, pt);
		if(Double.isInfinite(tree.dist_metric.getP()))
			return d;
		return Math.pow(d, 1d / tree.dist_metric.getP());
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d_lo, d_hi, d, rdist = 0.0;
		
		for(int j = 0; j < N_FEATURES; j++) {
			d_lo = tree.node_bounds[0][i_node][j] - pt[j];
			d_hi = pt[j] - tree.node_bounds[1][i_node][j];
			d = (d_lo + FastMath.abs(d_lo)) + (d_hi	+ FastMath.abs(d_hi));
			
			if(tree.dist_metric.getP() == Double.POSITIVE_INFINITY)
				rdist = FastMath.max(rdist, 0.5 * d);
			else
				rdist += FastMath.pow(0.5 * d, tree.dist_metric.getP());
		}
		
		return rdist;
	}

	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = maxRDist(tree, i_node, pt);
		if(Double.isInfinite(tree.dist_metric.getP()))
			return d;
		return Math.pow(d, 1d / tree.dist_metric.getP());
	}

	@Override
	double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d_lo, d_hi, rdist = 0.0;
		
		if(tree.dist_metric.getP() == Double.POSITIVE_INFINITY) {
			for(int j = 0; j < N_FEATURES; j++) {
				rdist = FastMath.max(rdist, FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]));
				rdist = FastMath.max(rdist, FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]));
			}
		} else {
			for(int j = 0; j < N_FEATURES; j++) {
				d_lo = FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]);
				d_hi = FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]);
				rdist += FastMath.pow(FastMath.max(d_lo, d_hi), tree.dist_metric.getP());
			}
		}
		
		return rdist;
	}

	@Override
	double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		double d1, d2, rdist = 0.0;
		int j;
		
		if(tree1.dist_metric.getP() == Double.POSITIVE_INFINITY) {
			for(j = 0; j < N_FEATURES; j++) {
				rdist = FastMath.max(rdist, 
					FastMath.abs(tree1.node_bounds[0][iNode1][j]
								- tree2.node_bounds[1][iNode2][j]));
				rdist = FastMath.max(rdist, 
						FastMath.abs(tree1.node_bounds[1][iNode1][j]
								- tree2.node_bounds[0][iNode2][j]));
			}
		} else {
			for(j = 0; j < N_FEATURES; j++) {
				d1 = FastMath.abs(tree1.node_bounds[0][iNode1][j]
								- tree2.node_bounds[1][iNode2][j]);
				d2 = FastMath.abs(tree1.node_bounds[1][iNode1][j]
								- tree2.node_bounds[0][iNode2][j]);
				rdist += FastMath.pow(FastMath.max(d1, d2), tree1.dist_metric.getP());
			}
		}
		
		return rdist;
	}

	@Override
	double maxDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.partialDistanceToDistance(maxRDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	double minDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.partialDistanceToDistance(minRDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	void minMaxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt, MutableDouble minDist, MutableDouble maxDist) {
		double d, d_lo, d_hi;
		int j;
		
		minDist.value = 0.0;
		maxDist.value = 0.0;
		
		if(tree.dist_metric.getP() == Double.POSITIVE_INFINITY) {
			for(j = 0; j < N_FEATURES; j++) {
				d_lo = tree.node_bounds[0][i_node][j] - pt[j];
				d_hi = pt[j] - tree.node_bounds[1][i_node][j];
				d = (d_lo + FastMath.abs(d_lo)) + (d_hi + FastMath.abs(d_hi));
				
				minDist.value = FastMath.max(minDist.value, 0.5 * d);
				maxDist.value = FastMath.max(maxDist.value, FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]));
				maxDist.value = FastMath.max(maxDist.value, FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]));
			}
		} else {
			for(j = 0; j < N_FEATURES; j++) {
				d_lo = tree.node_bounds[0][i_node][j] - pt[j];
				d_hi = pt[j] - tree.node_bounds[1][i_node][j];
				d = (d_lo + FastMath.abs(d_lo)) + (d_hi + FastMath.abs(d_hi));
				
				minDist.value = FastMath.pow(0.5 * d, tree.dist_metric.getP());
				maxDist.value = FastMath.pow(FastMath.max(FastMath.abs(d_lo), FastMath.abs(d_hi)), tree.dist_metric.getP());
			}
			
			double pow = 1.0 / tree.dist_metric.getP();
			minDist.value = FastMath.pow(minDist.value, pow);
			maxDist.value = FastMath.pow(maxDist.value, pow);
		} // end if/else
	}
}
