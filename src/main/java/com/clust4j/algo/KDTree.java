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
 * A <i>k</i>-d tree (short for k-dimensional tree) is a space-partitioning 
 * data structure for organizing points in a k-dimensional space. <i>k</i>-d 
 * trees are a useful data structure for several applications, such as searches 
 * involving a multidimensional search key (e.g. range searches and nearest 
 * neighbor searches). <i>k</i>-d trees are a special case of binary space partitioning trees.
 * @author Taylor G Smith
 * @see NearestNeighborHeapSearch
 * @see <a href="https://en.wikipedia.org/wiki/K-d_tree"><i>k</i>-d trees</a>
 */
public class KDTree extends NearestNeighborHeapSearch {
	private static final long serialVersionUID = -3744545394278454548L;
	public final static HashSet<Class<? extends GeometricallySeparable>> VALID_METRICS;
	static {
		VALID_METRICS = new HashSet<>();
		VALID_METRICS.add(Distance.EUCLIDEAN.getClass());
		VALID_METRICS.add(Distance.MANHATTAN.getClass());
		VALID_METRICS.add(MinkowskiDistance.class);
		VALID_METRICS.add(Distance.CHEBYSHEV.getClass());
	}
	
	
	@Override boolean checkValidDistMet(GeometricallySeparable dist) {
		return VALID_METRICS.contains(dist.getClass());
	}
	
	
	
	public KDTree(final RealMatrix X) {
		super(X);
	}
	
	public KDTree(final RealMatrix X, int leaf_size) {
		super(X, leaf_size);
	}
	
	public KDTree(final RealMatrix X, DistanceMetric dist) {
		super(X, dist);
	}
	
	public KDTree(final RealMatrix X, Loggable logger) {
		super(X, logger);
	}

	public KDTree(final RealMatrix X, int leaf_size, DistanceMetric dist) {
		super(X, leaf_size, dist);
	}
	
	public KDTree(final RealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	protected KDTree(final double[][] X, int leaf_size, DistanceMetric dist, Loggable logger) {
		super(X, leaf_size, dist, logger);
	}
	
	/**
	 * Constructor with logger and distance metric
	 * @param X
	 * @param dist
	 * @param logger
	 */
	public KDTree(final RealMatrix X, DistanceMetric dist, Loggable logger) {
		super(X, dist, logger);
	}
	
	
	
	@Override
	void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features) {
		tree.node_bounds = new double[2][n_nodes][n_features];
	}

	@Override
	void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end) {
		int n_features = tree.N_FEATURES, i, j;
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
			
			if( tree.infinity_dist )
				rad = FastMath.max(rad, 0.5 * (upperBounds[j] - lowerBounds[j]));
			else
				rad += FastMath.pow(
							0.5 * FastMath.abs(upperBounds[j] - lowerBounds[j]), 
							tree.dist_metric.getP());
		}
		
		tree.node_data[i_node].idx_start = idx_start;
		tree.node_data[i_node].idx_end = idx_end;
		
		// radius assignment
		tree.node_data[i_node].radius = Math.pow(rad, 1.0 / tree.dist_metric.getP());
	}

	@Override
	final KDTree newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger) {
		return new KDTree(new Array2DRowRealMatrix(arr, false), leaf, dist, logger);
	}
	
	@Override
	double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = minRDist(tree, i_node, pt);
		return tree.dist_metric.partialDistanceToDistance(d);
	}

	@Override
	double minDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.partialDistanceToDistance(minRDistDual(tree1, iNode1, tree2, iNode2));
	}

	@Override
	double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d_lo, d_hi, d, rdist = 0.0, p = tree.dist_metric.getP();
		final boolean inf = tree.infinity_dist;
		
		for(int j = 0; j < N_FEATURES; j++) {
			d_lo = tree.node_bounds[0][i_node][j] - pt[j];
			d_hi = pt[j] - tree.node_bounds[1][i_node][j];
			d = (d_lo + FastMath.abs(d_lo)) + (d_hi	+ FastMath.abs(d_hi));
			
			rdist = inf ? FastMath.max(rdist, 0.5 * d) :
				rdist + FastMath.pow(0.5 * d, p);
		}
		
		return rdist;
	}

	@Override
	double minRDistDual(NearestNeighborHeapSearch tree1, int i_node1, NearestNeighborHeapSearch tree2, int i_node2) {
		double d, d1, d2, rdist = 0.0, p = tree1.dist_metric.getP();
		int j, n_features = tree1.N_FEATURES;
		boolean inf = tree1.infinity_dist;
		
		for(j = 0; j < n_features; j++) {
			d1 = (tree1.node_bounds[0][i_node1][j] - tree2.node_bounds[1][i_node2][j]);
			d2 = (tree2.node_bounds[0][i_node2][j] - tree1.node_bounds[1][i_node1][j]);
			d  = (d1 + FastMath.abs(d1)) + (d2 + FastMath.abs(d2));
			rdist = inf ? FastMath.max(rdist, 0.5 * d) :
				rdist + FastMath.pow(0.5 * d, p);
		}
		
		return rdist;
	}

	@Override
	double maxDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		return tree1.dist_metric.partialDistanceToDistance(maxRDistDual(tree1, iNode1, tree2, iNode2));
	}

	/*
	@Override
	double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d = maxRDist(tree, i_node, pt);
		return tree.dist_metric.partialDistanceToDistance(d);
	}

	@Override
	double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] pt) {
		double d_lo, d_hi, rdist = 0.0, p = tree.dist_metric.getP();
		boolean inf = tree.infinity_dist;
		int n_features = tree.N_FEATURES;
		
		if(inf) {
			for(int j = 0; j < n_features; j++) {
				rdist = FastMath.max(rdist, FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]));
				rdist = FastMath.max(rdist, FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]));
			}
		} else {
			for(int j = 0; j < n_features; j++) {
				d_lo = FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]);
				d_hi = FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]);
				rdist += FastMath.pow(FastMath.max(d_lo, d_hi), p);
			}
		}
		
		return rdist;
	}
	*/

	@Override
	double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2) {
		double d1, d2, rdist = 0.0, p = tree1.dist_metric.getP();
		int j, n_features = tree1.N_FEATURES;
		final boolean inf = tree1.infinity_dist;
		
		if(inf) {
			for(j = 0; j < n_features; j++) {
				rdist = FastMath.max(rdist, 
					FastMath.abs(tree1.node_bounds[0][iNode1][j]
								- tree2.node_bounds[1][iNode2][j]));
				rdist = FastMath.max(rdist, 
						FastMath.abs(tree1.node_bounds[1][iNode1][j]
								- tree2.node_bounds[0][iNode2][j]));
			}
		} else {
			for(j = 0; j < n_features; j++) {
				d1 = FastMath.abs(tree1.node_bounds[0][iNode1][j]
								- tree2.node_bounds[1][iNode2][j]);
				d2 = FastMath.abs(tree1.node_bounds[1][iNode1][j]
								- tree2.node_bounds[0][iNode2][j]);
				rdist += FastMath.pow(FastMath.max(d1, d2), p);
			}
		}
		
		return rdist;
	}
	

	@Override
	void minMaxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt, MutableDouble minDist, MutableDouble maxDist) {
		double d, d_lo, d_hi, p = tree.dist_metric.getP();
		int j, n_features = tree.N_FEATURES;
		boolean inf = tree.infinity_dist;
		
		minDist.value = 0.0;
		maxDist.value = 0.0;
		
		for(j = 0; j < n_features; j++) {
			d_lo = tree.node_bounds[0][i_node][j] - pt[j];
			d_hi = pt[j] - tree.node_bounds[1][i_node][j];
			d = (d_lo + FastMath.abs(d_lo)) + (d_hi + FastMath.abs(d_hi));
			
			if( inf ) {
				minDist.value = FastMath.max(minDist.value, 0.5 * d);
				maxDist.value = FastMath.max(maxDist.value, 
											FastMath.abs(pt[j] - tree.node_bounds[0][i_node][j]));
				maxDist.value = FastMath.max(maxDist.value, 
											FastMath.abs(pt[j] - tree.node_bounds[1][i_node][j]));
			} else {
				minDist.value += FastMath.pow(0.5 * d, p);
				maxDist.value += FastMath.pow(
					FastMath.max(FastMath.abs(d_lo), FastMath.abs(d_hi)), p);
			}
		}
		
		
		if( !inf ) {
			double pow = 1.0 / p;
			minDist.value = FastMath.pow(minDist.value, pow);
			maxDist.value = FastMath.pow(maxDist.value, pow);
		}
	}
}
