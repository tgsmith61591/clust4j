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

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.QuadTup;
import com.clust4j.utils.VecUtils;

import static com.clust4j.GlobalState.Mathematics.*;

import java.util.Arrays;


/**
 * A datastructure for optimized high dimensional k-neighbors and radius
 * searches. Based on sklearns' BinaryTree class.
 * @author Taylor G Smith
 * @see <a href="https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi">sklearn BinaryTree</a>
 */
abstract class NearestNeighborHeapSearch implements java.io.Serializable {
	private static final long serialVersionUID = -5617532034886067210L;
	
	final static public int DEF_LEAF_SIZE = 40;
	final static public DistanceMetric DEF_DIST = Distance.EUCLIDEAN;
	final static String MEM_ERR = "Internal: memory layout is flawed: " +
		"not enough nodes allocated";
	
	
	
	double[][] data_arr;
	int[] idx_array;
	NodeData[] node_data;
	double[][][] node_bounds;
	
	/** If there's a logger, for warnings will issue warn message */
	final Loggable logger;
	/** Constrained to Dist, not Sim due to nearest neighbor requirements */
	final DistanceMetric dist_metric;
	int n_trims, n_leaves, n_splits, n_calls, leaf_size, n_levels, n_nodes;
	final int N_SAMPLES, N_FEATURES;
	/** Whether or not the algorithm uses the Inf distance, {@link Distance#CHEBYSHEV} */
	final boolean infinity_dist;
	
	
	
	/**
	 * Ensure valid metric
	 */
	abstract boolean checkValidDistMet(GeometricallySeparable dist);
	
	
	
	
	
	public NearestNeighborHeapSearch(final RealMatrix X) {
		this(X, DEF_LEAF_SIZE, DEF_DIST);
	}
	
	public NearestNeighborHeapSearch(final RealMatrix X, int leaf_size) {
		this(X, leaf_size, DEF_DIST);
	}
	
	public NearestNeighborHeapSearch(final RealMatrix X, DistanceMetric dist) {
		this(X, DEF_LEAF_SIZE, dist);
	}
	
	public NearestNeighborHeapSearch(final RealMatrix X, Loggable logger) {
		this(X, DEF_LEAF_SIZE, DEF_DIST, logger);
	}
	
	/**
	 * Default constructor without logger object
	 * @param X
	 * @param leaf_size
	 * @param dist
	 */
	public NearestNeighborHeapSearch(final RealMatrix X, int leaf_size, DistanceMetric dist) {
		this(X, leaf_size, dist, null);
	}
	
	/**
	 * Constructor with logger and distance metric
	 * @param X
	 * @param dist
	 * @param logger
	 */
	public NearestNeighborHeapSearch(final RealMatrix X, DistanceMetric dist, Loggable logger) {
		this(X, DEF_LEAF_SIZE, dist, logger);
	}
	
	/**
	 * Constructor with logger object
	 * @param X
	 * @param leaf_size
	 * @param dist
	 * @param logger
	 */
	public NearestNeighborHeapSearch(final RealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		this(X.getData(), leaf_size, dist, logger);
	}
	
	/**
	 * Constructor with logger object
	 * @param X
	 * @param leaf_size
	 * @param dist
	 * @param logger
	 */
	protected NearestNeighborHeapSearch(final double[][] X, int leaf_size, DistanceMetric dist, Loggable logger) {
		this.data_arr = MatUtils.copy(X);
		this.leaf_size = leaf_size;
		this.logger = logger;
		
		if(leaf_size < 1)
			throw new IllegalArgumentException("illegal leaf size: " + leaf_size);
		
		if(!checkValidDistMet(dist)) {
			if(null != logger)
				logger.warn(dist+" is not valid for " + this.getClass() +". Reverting to " + DEF_DIST);
			this.dist_metric = DEF_DIST;
		} else {
			this.dist_metric = dist;
		}
		
		
		// Whether the algorithm is using the infinity distance (Chebyshev)
		this.infinity_dist = this.dist_metric.getP() == Double.POSITIVE_INFINITY ||
			Double.isInfinite(this.dist_metric.getP());
		
		
		// determine number of levels in the tree, and from this
        // the number of nodes in the tree.  This results in leaf nodes
        // with numbers of points between leaf_size and 2 * leaf_size
		MatUtils.checkDims(this.data_arr);
		N_SAMPLES = data_arr.length;  
		N_FEATURES = X[0].length;
		
		/*
		// Should round up or always take floor function?...
		double nlev = FastMath.log(2, FastMath.max(1, (N_SAMPLES-1)/leaf_size)) + 1;
		this.n_levels = (int)FastMath.round(nlev);
		this.n_nodes = (int)(FastMath.pow(2, nlev) - 1);
		*/
		
		this.n_levels = (int)(FastMath.log(2, FastMath.max(1, (N_SAMPLES-1)/leaf_size)) + 1);
		this.n_nodes = (int)(FastMath.pow(2, n_levels) - 1);
	
		// allocate arrays for storage
		this.idx_array = VecUtils.arange(N_SAMPLES);
		
		// Add new NodeData objs to node_data arr
		this.node_data = new NodeData[n_nodes];
		for(int i = 0; i < node_data.length; i++)
			node_data[i] = new NodeData();
		
		// allocate tree specific data
		allocateData(this, n_nodes, N_FEATURES);
		recursiveBuild(0, 0, N_SAMPLES);
	}
	
	
	
	
	
	// ========================== Inner classes ==========================
	
	interface Density {
		double getDensity(double dist, double h);
		double getNorm(double h, int d);
	}
	
	/**
	 * Provides efficient, reduced kernel approximations for points
	 * that are faster and simpler than the {@link Kernel} class methods.
	 * @author Taylor G Smith
	 */
	public static enum PartialKernelDensity implements Density, java.io.Serializable {
		LOG_COSINE {
			@Override
			public double getDensity(double dist, double h) {
				return dist < h ? FastMath.log(FastMath.cos(0.5 * Math.PI * dist / h)) : Double.NEGATIVE_INFINITY;
			}
			
			@Override
			public double getNorm(double h, int d) {
				double factor = 0;
				double tmp = 2d / Math.PI;
				
				for(int k = 1; k < d + 1; k += 2) {
					factor += tmp;
					tmp *= -(d - k) * (d - k - 1) * FastMath.pow((2.0 / Math.PI), 2);
				}
				
				return FastMath.log(factor) + logSn(d - 1);
			}
		},
		
		LOG_EPANECHNIKOV {
			@Override
			public double getDensity(double dist, double h) {
				return dist < h ? FastMath.log(1.0 - (dist * dist)/(h * h)) : Double.NEGATIVE_INFINITY;
			}
			
			@Override
			public double getNorm(double h, int d) {
				return logVn(d) + FastMath.log(2.0 / (d + 2.0));
			}
		},
		
		LOG_EXPONENTIAL {
			@Override
			public double getDensity(double dist, double h) {
				return -dist / h;
			}
			
			@Override
			public double getNorm(double h, int d) {
				return logSn(d - 1) + lgamma(d);
			}
		},
		
		LOG_GAUSSIAN {
			@Override
			public double getDensity(double dist, double h) {
				return -0.5 * (dist * dist) / (h * h);
			}
			
			@Override
			public double getNorm(double h, int d) {
				return 0.5 * d * LOG_2PI;
			}
		},
		
		LOG_LINEAR {
			@Override
			public double getDensity(double dist, double h) {
				return dist < h ? FastMath.log(1 - dist / h) : Double.NEGATIVE_INFINITY;
			}
			
			@Override
			public double getNorm(double h, int d) {
				return logVn(d) - FastMath.log(d + 1.0);
			}
		},
		
		LOG_TOPHAT {
			@Override
			public double getDensity(double dist, double h) {
				return dist < h ? 0 : Double.NEGATIVE_INFINITY;
			}
			
			@Override
			public double getNorm(double h, int d) {
				return logVn(d);
			}
		}
	}
	
	
	
	/**
	 * A hacky container for passing double references...
	 * Allows us to modify the value of a double as if
	 * we had passed a pointer. Since much of this code
	 * is adapted from Pyrex, Cython and C code, it
	 * leans heavily on passing pointers.
	 * @author Taylor G Smith
	 */
	// Tested: passing
	public static class MutableDouble implements Comparable<Double>, java.io.Serializable {
		private static final long serialVersionUID = -4636023903600763877L;
		public Double value = new Double(0);
		
		MutableDouble() { }
		MutableDouble(Double value) {
			this.value = value;
		}
		
		@Override
		public int compareTo(final Double n) {
			return value.compareTo(n);
		}
	}
	
	/**
	 * Node data container
	 * @author Taylor G Smith
	 */
	// Tested: passing
	public static class NodeData implements DeepCloneable, java.io.Serializable {
		private static final long serialVersionUID = -2469826821608908612L;
		int idx_start, idx_end;
		boolean is_leaf;
		double radius;
		
		public NodeData() { }
		public NodeData(int st, int ed, boolean is, double rad) {
			idx_start = st;
			idx_end = ed;
			is_leaf = is;
			radius = rad;
		}
		
		@Override
		public String toString() {
			return "NodeData: ["+idx_start+", "+
				idx_end+", "+is_leaf+", "+radius+"]";
		}
		
		@Override
		public NodeData copy() {
			return new NodeData(idx_start, idx_end, is_leaf, radius);
		}
		
		@Override
		public boolean equals(Object o) {
			if(this == o)
				return true;
			if(o instanceof NodeData) {
				NodeData nd = (NodeData)o;
				return nd.idx_start == this.idx_start
					&& nd.idx_end   == this.idx_end
					&& nd.is_leaf   == this.is_leaf
					&& nd.radius    == this.radius;
			}
			
			return false;
		}
		
		public boolean isLeaf() {
			return is_leaf;
		}
		
		public int end() {
			return idx_end;
		}
		
		public double radius() {
			return radius;
		}
		
		public int start() {
			return idx_start;
		}
	}
	
	/**
	 * Abstract super class for NodeHeap and
	 * NeighborHeap classes
	 * @author Taylor G Smith
	 */
	abstract static class Heap implements java.io.Serializable {
		private static final long serialVersionUID = 8073174366388667577L;

		abstract static class Node {
			double val;
			int i1;
			int i2;
			
			Node() {}
			Node(double val, int i1, int i2) {
				this.val = val;
				this.i1 = i1;
				this.i2 = i2;
			}
		}
		
		Heap(){}
		
		static void swapNodes(Node[] arr, int i1, int i2) {
			Node tmp = arr[i1];
			arr[i1] = arr[i2];
			arr[i2] = tmp;
		}
		
		static void dualSwap(double[] darr, int[] iarr, int i1, int i2) {
			final double dtmp = darr[i1];
			darr[i1] = darr[i2];
			darr[i2] = dtmp;
			
			final int itmp = iarr[i1];
			iarr[i1] = iarr[i2];
			iarr[i2] = itmp;
		}
	}
	
	/**
	 * A max-heap structure to keep track of distances/indices of neighbors
     * This is based on the sklearn.neighbors.binary_tree module's NeighborsHeap class
     * 
	 * @author Taylor G Smith, adapted from sklearn
	 * @see <a href="https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi">sklearn NodeHeap</a>
	 */
	static class NeighborsHeap extends Heap {
		private static final long serialVersionUID = 3065531260075044616L;
		double[][] distances;
		int[][] indices;
		
		NeighborsHeap(int nPts, int k) {
			super();
			distances = MatUtils.rep(Double.POSITIVE_INFINITY, nPts, k);
			indices   = new int[nPts][k];
		}
		
		Neighborhood getArrays(boolean sort) {
			if(sort)
				this.sort();
			return new Neighborhood(distances, indices);
		}
		
		int push(int row, double val, int i_val) {
			int i, ic1, ic2, i_swap, size = distances[0].length;
			double[] dist_arr = distances[row];
			int[] ind_arr = indices[row];
			
			if(val > dist_arr[0])
				return 0;
			
			// Insert at pos 0
			dist_arr[0] = val;
			ind_arr[0] = i_val;
			
			// Descend heap, swap vals until max heap criteria met
			i = 0;
			while(true) {
				ic1 = 2 * i + 1;
				ic2 = ic1 + 1;
				
				if(ic1 >= size)
					break;
				else if(ic2 >= size) {
					if(dist_arr[ic1] > val)
						i_swap = ic1;
					else
						break;
				} else if(dist_arr[ic1] >= dist_arr[ic2]) {
					if(val < dist_arr[ic1])
						i_swap = ic1;
					else
						break;
				} else {
					if(val < dist_arr[ic2])
						i_swap = ic2;
					else
						break;
				}
				
				dist_arr[i] = dist_arr[i_swap];
				ind_arr[i] = ind_arr[i_swap];
				
				i = i_swap;
			}
			
			dist_arr[i] = val;
			ind_arr[i] = i_val;
			
			return 0;
		}
		
		int sort() {
			for(int row = 0; row < distances.length; row++) {
				simultaneous_sort(
					this.distances[row], 
					this.indices[row], 
					distances[row].length);
			}
			
			return 0;
		}
		
		double largest(int row) {
			return distances[row][0];
		}
		
		static int simultaneous_sort(double[] dist, int[] idx, int size) {
			int pivot_idx, i, store_idx;
			double pivot_val;
			
			if(size <= 1){ // pass
			} 
			
			else if(size == 2) {
				if(dist[0] > dist[1])
					dualSwap(dist, idx, 0, 1);
			}
			
			/*
			else {
				int[] order = VecUtils.argSort(dist);
				dualOrderInPlace(dist, idx, order);
			}
			*/
			
			else if(size == 3) {
				if(dist[0] > dist[1])
					dualSwap(dist, idx, 0, 1);
				
				if(dist[1] > dist[2]) {
					dualSwap(dist, idx, 1, 2);
					if(dist[0] > dist[1])
						dualSwap(dist, idx, 0, 1);
				}
			} 
			
			else {
				pivot_idx = size / 2;
				if(dist[0] > dist[size - 1])
					dualSwap(dist, idx, 0, size - 1);
				
				if(dist[size - 1] > dist[pivot_idx]) {
					dualSwap(dist, idx, size - 1, pivot_idx);
					if(dist[0] > dist[size - 1])
						dualSwap(dist, idx, 0, size - 1);
				}
				pivot_val = dist[size - 1];
				
				store_idx = 0;
				for(i = 0; i < size - 1; i++) {
					if(dist[i] < pivot_val) {
						dualSwap(dist, idx, i, store_idx);
						store_idx++;
					}
				}
				
				dualSwap(dist, idx, store_idx, size - 1);
				pivot_idx = store_idx;
				
				if(pivot_idx > 1)
					simultaneous_sort(dist, idx, pivot_idx);
				
				if(pivot_idx + 2 < size) {
					// Can't pass reference to middle of array, so sort copy
					// and then iterate over sorted, replacing in place
					final int sliceStart = pivot_idx + 1;
					final int sliceEnd = dist.length;
					
					final int newLen = sliceEnd - sliceStart;
					double[] a = new double[newLen];
					int[] b = new int[newLen];
					
					System.arraycopy(dist, sliceStart, a, 0, newLen);
					System.arraycopy(idx, sliceStart, b, 0, newLen);
					
					simultaneous_sort(a, b, size - pivot_idx - 1);
					
					
					// Now iter over and replace...
					for(int k = 0, p = sliceStart; p < sliceEnd; k++, p++) {
						dist[p] = a[k];
						idx[p]  = b[k];
					}
				}
			}
			
			return 0;
		}
	}
	
	/**
	 * A min heap implementation for keeping track of nodes
	 * during a breadth first search. This is based on the
	 * sklearn.neighbors.binary_tree module's NodeHeap class.
	 * 
	 * <p>
	 * Internally, the data is stored in a simple binary 
	 * heap which meetsthe min heap condition:
	 * 
	 * <p>
	 * <tt>heap[i].val < min(heap[2 * i + 1].val, heap[2 * i + 2].val)</tt>
	 * 
	 * @author Taylor G Smith, adapted from sklearn
	 * @see <a href="https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi">sklearn NodeHeap</a>
	 */
	static class NodeHeap extends Heap {
		private static final long serialVersionUID = 5621403002445703132L;
		NodeHeapData[] data;
		int n;
		
		/** Node class. */
		static class NodeHeapData extends Node {
			NodeHeapData() { super(); }
			NodeHeapData(double val, int i1, int i2) {
				super(val, i1, i2);
			}
			
			@Override
			public boolean equals(Object o) {
				if(o == this)
					return true;
				if(o instanceof NodeHeapData) {
					NodeHeapData n = (NodeHeapData)o;
					return n.val == this.val
						&& n.i1 == this.i1
						&& n.i2 == this.i2;
				}
				
				return false;
			}
			
			@Override
			public String toString() {
				return "{" + val + ", " + i1 + ", " + i2 + "}";
 			}
		}
		
		NodeHeap(int size) {
			super();
			size = FastMath.max(size, 1);
			data = new NodeHeapData[size];
			
			//n = size;
			clear();
		}
		
		void clear() {
			n = 0;
		}
		
		NodeHeapData peek() {
			return data[0];
		}
		
		/**
		 * Remove and return first element in heap
		 * @return
		 */
		NodeHeapData pop() {
			if(this.n == 0)
				throw new IllegalStateException("cannot pop an empty heap");
			
			int i, i_child1, i_child2, i_swap;
			NodeHeapData popped_element = this.data[0];
			
			// pop off the first element, move the last element to the front,
	        // and then perform swaps until the heap is back in order
			this.data[0] = this.data[this.n - 1];
			
			// Omitted from sklearn, but added here; make last element null again...
			this.data[this.n - 1] = null;
			this.n--;
			
			i = 0;
			
			while(i < this.n) {
				i_child1 = 2 * i + 1;
				i_child2 = 2 * i + 2;
				i_swap = 0;
				
				
				if(i_child2 < this.n) {
					if(this.data[i_child1].val <= this.data[i_child2].val)
						i_swap = i_child1;
					else
						i_swap = i_child2;
				} else if(i_child1 < this.n) {
					i_swap = i_child1;
				} else {
					break;
				}
				
				
				if(i_swap > 0 && this.data[i_swap].val <= this.data[i].val) {
					swapNodes(this.data, i, i_swap);
					i = i_swap;
				} else {
					break;
				}
			}
			
			return popped_element;
		}
		
		int push(NodeHeapData node) {
			// Add to the heap
			int i;
			this.n++;
			
			// If the new n exceeds current length,
			// double the size of the data array
			if(this.n > this.data.length)
				resize(2 * this.n);
			
			// Put new element at end, perform swaps
			i = this.n - 1;
			this.data[i] = node;
			reorderFromPush(i);
			
			return 0;
		}
		
		private void reorderFromPush(int i) {
			int i_parent;
			while(i > 0) {
				i_parent = (i - 1) / 2;
				if(this.data[i_parent].val <= this.data[i].val)
					break;
				else {
					swapNodes(this.data, i, i_parent);
					i = i_parent;
				}
			}
		}
		
		int resize(int new_size) {
			if(new_size < 1)
				throw new IllegalArgumentException("cannot resize heap "
						+ "to size less than 1 (" + new_size + ")");
			
			// Resize larger or smaller
			int size = this.data.length;
			final int oldN = n;
			NodeHeapData[] newData = new NodeHeapData[new_size];
			
			// Original sklearn line included if clause, but due to our
			// new IAE check, we can skip it and enter for loop automatically:
			// if(size > 0 && new_size > 0)
			
			for(int i = 0; i < FastMath.min(size, new_size); i++)
				newData[i] = this.data[i];
			
			// Original sklearn line, but seems to be
			// buggy. n is supposed to be count of objs inside,
			// and as it stands, this makes n equal to total size
			// of the heap.
			/*
			if(new_size < size)
				this.n = new_size;
			*/
			
			// New line that accts for above corner case:
			if(new_size < size)
				this.n = FastMath.min(new_size, oldN);
			
			this.data = newData;
			return 0;
		}
		
		@Override
		public String toString() {
			return Arrays.toString(this.data);
		}
	}
	
	
	
	
	
	
	
	// ========================== Getters ==========================
	public double[][] getData() {
		return MatUtils.copy(data_arr);
	}
	
	double[][] getDataRef() {
		return data_arr;
	}
	
	public int getLeafSize() {
		return leaf_size;
	}
	
	public DistanceMetric getMetric() {
		return dist_metric;
	}
	
	public double[][][] getNodeBounds() {
		int m = node_bounds.length;
		
		double[][][] out = new double[m][][];
		for(int i = 0; i < m; i++)
			out[i] = MatUtils.copy(node_bounds[i]);
		
		return out;
	}
	
	double[][][] getNodeBoundsRef() {
		return node_bounds;
	}
	
	public int[] getIndexArray() {
		return VecUtils.copy(idx_array);
	}
	
	int[] getIndexArrayRef() {
		return idx_array;
	}
	
	public NodeData[] getNodeData() {
		NodeData[] copy = new NodeData[node_data.length];
		for(int i = 0; i < copy.length; i++)
			copy[i] = node_data[i].copy();
		return copy;
	}
	
	NodeData[] getNodeDataRef() {
		return node_data;
	}
	
	
	// ========================== Instance methods ==========================
	double dist(final double[] a, final double[] b) {
		n_calls++;
		return dist_metric.getDistance(a, b);
	}
	
	public int getNumCalls() {
		return n_calls;
	}
	
	double rDist(final double[] a, final double[] b) {
		n_calls++;
		return dist_metric.getPartialDistance(a, b);
	}
	
	double rDistToDist(final double d) {
		return dist_metric.partialDistanceToDistance(d);
	}
	
	private void rDistToDistInPlace(final double[][] d) {
		final int m = d.length, n = d[0].length;
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				d[i][j] = rDistToDist(d[i][j]);
	}
	
	private void estimateKernelDensitySingleDepthFirst(int i_node, double[] pt, PartialKernelDensity kern, double h,
			double logKNorm, double logAbsTol, double logRelTol, double localLogMinBound, double localLogBoundSpread,
			MutableDouble globalLogMinBound, MutableDouble globalLogBoundSpread) {
		
		int i, i1, i2, N1, N2;
		double[][] data = this.data_arr;
		NodeData nodeInfo = this.node_data[i_node];
		double dist_pt, logDensContribution;
		
		double child1LogMinBound, child2LogMinBound, child1LogBoundSpread, child2LogBoundSpread;
		MutableDouble dist_UB = new MutableDouble(), dist_LB = new MutableDouble();
		
		N1 = nodeInfo.idx_end - nodeInfo.idx_start;
		N2 = N_SAMPLES;
		double logN1 = FastMath.log(N1), logN2 = FastMath.log(N2);
		
		// If local bounds equal to within errors
		if(logKNorm + localLogBoundSpread - logN1 + logN2
			<= logAddExp(logAbsTol, (logRelTol + logKNorm + localLogMinBound))) {
			return;
		}
		
		// If global bounds are within rel tol & abs tol
		else if(logKNorm + globalLogBoundSpread.value
			<= logAddExp(logAbsTol, (logRelTol + logKNorm + globalLogMinBound.value))) {
			return;
		}
		
		// node is a leaf
		else if(nodeInfo.is_leaf) {
			globalLogMinBound.value = logSubExp(globalLogMinBound.value, localLogMinBound);
			globalLogBoundSpread.value = logSubExp(globalLogBoundSpread.value, localLogBoundSpread);
			
			for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
				dist_pt = this.dist(pt, data[idx_array[i]]);
				logDensContribution = kern.getDensity(dist_pt, h);
				globalLogMinBound.value = logAddExp(globalLogMinBound.value, logDensContribution);
			}
		}
		
		// Split and query
		else {
			i1 = 2 * i_node + 1;
			i2 = 2 * i_node + 2;
			
			N1 = this.node_data[i1].idx_end - this.node_data[i1].idx_start;
			N2 = this.node_data[i2].idx_end - this.node_data[i2].idx_start;
			logN1 = FastMath.log(N1); 
			logN2 = FastMath.log(N2);
			
			// Mutates distLB & distUB internally
			minMaxDist(this, i1, pt, dist_LB, dist_UB);
			child1LogMinBound = logN1 + kern.getDensity(dist_UB.value, h);
			child1LogBoundSpread = logSubExp(logN1 + kern.getDensity(dist_LB.value, h), child1LogMinBound);

			// Mutates distLB & distUB internally
			minMaxDist(this, i2, pt, dist_LB, dist_UB);
			child2LogMinBound = logN2 + kern.getDensity(dist_UB.value, h);
			child2LogBoundSpread = logSubExp(logN2 + kern.getDensity(dist_LB.value, h), child2LogMinBound);
			
			// Update log min bound
			globalLogMinBound.value = logSubExp(globalLogMinBound.value, localLogMinBound);
			globalLogMinBound.value = logAddExp(globalLogMinBound.value, child1LogMinBound);
			globalLogMinBound.value = logAddExp(globalLogMinBound.value, child2LogMinBound);
			
			// Update log bound spread
			globalLogBoundSpread.value = logSubExp(globalLogBoundSpread.value, localLogBoundSpread);
			globalLogBoundSpread.value = logAddExp(globalLogBoundSpread.value, child1LogBoundSpread);
			globalLogBoundSpread.value = logAddExp(globalLogBoundSpread.value, child2LogBoundSpread);
			
			// Recurse left then right
			estimateKernelDensitySingleDepthFirst(i1, pt, kern, h, logKNorm,
					logAbsTol, logRelTol, child1LogMinBound, child1LogBoundSpread,
					globalLogMinBound, globalLogBoundSpread);
			
			estimateKernelDensitySingleDepthFirst(i2, pt, kern, h, logKNorm,
					logAbsTol, logRelTol, child2LogMinBound, child2LogBoundSpread,
					globalLogMinBound, globalLogBoundSpread);
		}
	}
	
	
	
	// Tested: passing
	public static int findNodeSplitDim(double[][] data, int[] idcs) {
		// Gets the difference between the vector of column
		// maxes and the vector of column mins, then finds the
		// arg max.
		
		// computes equivalent of (sklearn): 
		// j_max = np.argmax(np.max(data, 0) - np.min(data, 0))
		int n = data[0].length, idx, argMax = -1;
		double[] maxVec= VecUtils.rep(Double.NEGATIVE_INFINITY, n), 
				minVec = VecUtils.rep(Double.POSITIVE_INFINITY, n),
				current;
		double diff, maxDiff = Double.NEGATIVE_INFINITY;
		
		// Optimized to one KxN pass
		for(int i = 0; i < idcs.length; i++) {
			idx = idcs[i];
			current = data[idx];
			
			for(int j = 0; j < n; j++) {
				if(current[j] > maxVec[j])
					maxVec[j] = current[j];
				if(current[j] < minVec[j])
					minVec[j] = current[j];
				
				// If the last iter, we can calc difference right now
				if(i == idcs.length-1) {
					diff = maxVec[j] - minVec[j];
					if(diff > maxDiff) {
						maxDiff = diff;
						argMax = j;
					}
				}
			}
		}
		
		return argMax;
	}
	
	/**
	 * Returns a QuadTup with references to the arrays
	 * @return
	 */
	public QuadTup<double[][], int[], NodeData[], double[][][]> getArrays() {
		return new QuadTup<>(data_arr, idx_array, node_data, node_bounds);
	}
	
	public Triple<Integer, Integer, Integer> getTreeStats() {
		return new ImmutableTriple<>(n_trims, n_leaves, n_splits);
	}
	
	public double[] kernelDensity(double[][] X, double bandwidth, PartialKernelDensity kern, 
			double absTol, double relTol, boolean returnLog) {
		
		double b_c = bandwidth, logAbsTol = FastMath.log(absTol), 
				logRelTol = FastMath.log(relTol); 
		
		MutableDouble logMinBound = new MutableDouble(), 
				logMaxBound = new MutableDouble(), 
				logBoundSpread = new MutableDouble();
		MutableDouble dist_LB = new MutableDouble(), dist_UB = new MutableDouble();
		int m = data_arr.length, n = data_arr[0].length, i;
		
		
		// Ensure X col dim matches training data col dim
		MatUtils.checkDims(X);
		if(X[0].length != n)
			throw new DimensionMismatchException(n, X[0].length);

		
		final double logKNorm = logKernelNorm(b_c, n, kern), 
				logM = FastMath.log(m), log2 = FastMath.log(2);
		double[][] Xarr = MatUtils.copy(X);
		double[] logDensity = new double[Xarr.length], pt;
		
		for(i = 0; i < Xarr.length; i++) {
			pt = Xarr[i];
			
			minMaxDist(this, 0, pt, dist_LB, dist_UB);
			logMinBound.value = logM + kern.getDensity(dist_UB.value, b_c);
			logMaxBound.value = logM + kern.getDensity(dist_LB.value, b_c);
			logBoundSpread.value = logSubExp(logMaxBound.value, logMinBound.value);
			
			estimateKernelDensitySingleDepthFirst(0, pt, kern, b_c, logKNorm, 
					logAbsTol, logRelTol, logMinBound.value, logBoundSpread.value, 
					logMinBound, logBoundSpread);
			
			logDensity[i] = logAddExp(logMinBound.value, logBoundSpread.value - log2);
		}
		
		// Norm results
		for(i = 0; i < logDensity.length; i++)
			logDensity[i] += logKNorm;
		
		return returnLog ? logDensity : VecUtils.exp(logDensity);
	}
	
	private double logAddExp(double x1, double x2) {
		final double a = FastMath.max(x1, x2);
		if(Double.NEGATIVE_INFINITY == a)
			return a;
		return a + FastMath.log(FastMath.exp(x1 - a) + FastMath.exp(x2 - a));
	}
	
	static double logKernelNorm(double h, int i, PartialKernelDensity kern) {
		return -kern.getNorm(h, i) - i * FastMath.log(h);
	}
	
	static double logSn(int n) {
		return LOG_2PI + logVn(n - 1);
	}
	
	private double logSubExp(double x1, double x2) {
		if(x1 <= x2)
			return Double.NEGATIVE_INFINITY;
		return x1 + FastMath.log(1 - FastMath.exp(x2 - x1));
	}
	
	static double logVn(int n) {
		return 0.5 * n * LOG_PI - lgamma(0.5 * n + 1);
	}
	
	public static void partitionNodeIndices(double[][] data,
			int[] nodeIndices, int splitDim, int splitIndex,
			int nFeatures, int nPoints) {
			 
		int left = 0;
		int right = nPoints - 1;
		double d1, d2;
		
		while(true) {
			int midindex = left;
			
			for(int i = left; i < right; i++) {
				d1 = data[nodeIndices[i]][splitDim];
				d2 = data[nodeIndices[right]][splitDim];
				            
				if(d1 < d2) {
					swap(nodeIndices, i, midindex);
					midindex++;
				}
			}
			
			swap(nodeIndices, midindex, right);
			if(midindex == splitIndex) {
				break;
			} else if(midindex < splitIndex) {
				left = midindex + 1;
			} else {
				right = midindex - 1;
			}
		}
	}


	
	void resetNumCalls() {
		n_calls = 0;
	}
	
	void recursiveBuild(int i_node, int idx_start, int idx_end) {
		int i_max,
			n_points = idx_end - idx_start,
			n_mid = n_points / 2;
		initNode(this, i_node, idx_start, idx_end);
		
		
		if(2 * i_node + 1 >= this.n_nodes) {
			node_data[i_node].is_leaf = true;
			
			if(idx_end - idx_start > 2 * leaf_size) {
				if(null != logger)
					logger.warn(MEM_ERR);
			} else {/*really should only hit this block*/}
			
		} else if(idx_end - idx_start < 2) {
			if(null != logger)
				logger.warn(MEM_ERR);
			node_data[i_node].is_leaf = true;
		} else {
			// split node and recursively build child nodes
			node_data[i_node].is_leaf = false;
			i_max = findNodeSplitDim(data_arr, idx_array);
			partitionNodeIndices(data_arr, idx_array, 
					i_max, n_mid, N_FEATURES, n_points);
			
			recursiveBuild(2 * i_node + 1, idx_start, idx_start + n_mid);
			recursiveBuild(2 * i_node + 2, idx_start + n_mid, idx_end);
		}
	}
	
	/**
	 * Swap two indices in place
	 * @param idcs
	 * @param i1
	 * @param i2
	 */
	static void swap(int[] idcs, int i1, int i2) {
		int tmp = idcs[i1];
		idcs[i1] = idcs[i2];
		idcs[i2] = tmp;
	}
	
	/**
	 * Default query, which calls {@link #query(double[][], int, boolean, boolean)}
	 * <tt>(X, 1, false, true)</tt>
	 * @param X
	 * @return the neighborhood
	 */
	public Neighborhood query(double[][] X) {
		return query(X, 1, false, true);
	}
	
	
	public Neighborhood query(double[][] X, int k, boolean dualTree, boolean sort) {
		MatUtils.checkDims(X);
		
		final int n = data_arr[0].length, mPrime = X.length;
		
		
		if(n != X[0].length)
			throw new DimensionMismatchException(n, X[0].length);
		if(this.N_SAMPLES < k) 
			throw new IllegalArgumentException(k+" is greater than rows in data");
		if(k < 1) throw new IllegalArgumentException(k+" must exceed 0");
		
		
		double[][] Xarr = X;
		
		// Initialize neighbor heap
		NeighborsHeap heap = new NeighborsHeap(mPrime, k);
		
		double[] bounds, pt;
		double reduced_dist_LB;
		
		this.n_trims  = 0;
		this.n_leaves = 0;
		this.n_splits = 0;
		
		if(dualTree) {
			NearestNeighborHeapSearch other = newInstance(Xarr, leaf_size, dist_metric, logger);
			
			reduced_dist_LB = minRDistDual(this, 0, other, 0);
			bounds = VecUtils.rep(Double.POSITIVE_INFINITY, this.N_SAMPLES);
			queryDualDepthFirst(0, other, 0, bounds, heap, reduced_dist_LB);
		} else {
			int i;
			
			for(i = 0; i < mPrime; i++) {
				pt = Xarr[i];
				reduced_dist_LB = minRDist(this, 0, pt);
				querySingleDepthFirst(0, pt, i, heap, reduced_dist_LB);
			}
		}
		
		Neighborhood distances_indices = heap.getArrays(sort);
		int[][] indices = distances_indices.getValue();
		double[][] distances = distances_indices.getKey();
		rDistToDistInPlace(distances); // set back to dist
		
		return new Neighborhood(distances, indices);
	}
	
	private void queryDualDepthFirst(int i_node1, NearestNeighborHeapSearch other,
									 int i_node2, double[] bounds, NeighborsHeap heap,
									 double reduced_dist_LB) {
		NodeData node_info1 = this.node_data[i_node1],
				 node_info2 = other.node_data[i_node2];
		double[][] data1 = this.data_arr, data2 = other.data_arr;
		int i1, i2, i_pt, i_parent;
		double bound_max, dist_pt, reduced_dist_LB1, reduced_dist_LB2;
		
		
		// If nodes are farther apart than current bound
		if(reduced_dist_LB > bounds[i_node2]) { // Pass here
		} 

		// If both nodes are leaves
		else if(node_info1.is_leaf && node_info2.is_leaf) {
			bounds[i_node2] = 0;
			
			
			for(i2 = node_info2.idx_start; i2 < node_info2.idx_end; i2++) {
				i_pt = other.idx_array[i2];
				
				if(heap.largest(i_pt) <= reduced_dist_LB)
					continue;
				
				for(i1 = node_info1.idx_start; i1 < node_info1.idx_end; i1++) {
					
					// sklearn line:
					// data1 + n_features * self.idx_array[i1],
                    // data2 + n_features * i_pt
					dist_pt = rDist(data1[idx_array[i1]], data2[i_pt]);
					if(dist_pt < heap.largest(i_pt))
						heap.push(i_pt, dist_pt, idx_array[i1]);
				}
				
				// Keep track of node bound
				bounds[i_node2] = FastMath.max(bounds[i_node2], 
									heap.largest(i_pt));
			}
			
			
			// Update bounds
			while(i_node2 > 0) {
				i_parent = (i_node2 - 1) / 2;
				bound_max = FastMath.max(bounds[2 * i_parent + 1], 
									     bounds[2 * i_parent + 2]);
				if(bound_max < bounds[i_parent]) {
					bounds[i_parent] = bound_max;
					i_node2 = i_parent;
				} else break;
			}
		}
		
		// When node 1 is a leaf or is smaller
		else if(node_info1.is_leaf 
				|| (!node_info2.is_leaf
					&& node_info2.radius > node_info1.radius)) {
			
			reduced_dist_LB1 = minRDistDual(this, i_node1, other, 2 * i_node2 + 1);
			reduced_dist_LB2 = minRDistDual(this, i_node1, other, 2 * i_node2 + 2);
			
			if(reduced_dist_LB1 < reduced_dist_LB2) {
				queryDualDepthFirst(i_node1, other, 2 * i_node2 + 1, bounds, heap, reduced_dist_LB1);
				queryDualDepthFirst(i_node1, other, 2 * i_node2 + 2, bounds, heap, reduced_dist_LB2);
			} else { 
				// Do it in the opposite order...
				queryDualDepthFirst(i_node1, other, 2 * i_node2 + 2, bounds, heap, reduced_dist_LB2);
				queryDualDepthFirst(i_node1, other, 2 * i_node2 + 1, bounds, heap, reduced_dist_LB1);
			}
		}
		
		// Otherwise node 2 is a leaf or is smaller
		else {
			reduced_dist_LB1 = minRDistDual(this, 2 * i_node1 + 1, other, i_node2);
			reduced_dist_LB2 = minRDistDual(this, 2 * i_node1 + 2, other, i_node2);
			
			if(reduced_dist_LB1 < reduced_dist_LB2) {
				queryDualDepthFirst(2 * i_node1 + 1, other, i_node2, bounds, heap, reduced_dist_LB1);
				queryDualDepthFirst(2 * i_node1 + 2, other, i_node2, bounds, heap, reduced_dist_LB2);
			} else {
				// Do it in the opposite order...
				queryDualDepthFirst(2 * i_node1 + 2, other, i_node2, bounds, heap, reduced_dist_LB2);
				queryDualDepthFirst(2 * i_node1 + 1, other, i_node2, bounds, heap, reduced_dist_LB1);
			}
		}
	}
	
	private void ensurePositiveRadius(final double radius) {
		RadiusNeighbors.validateRadius(radius);
	}
	
	public Neighborhood queryRadius(final RealMatrix X, double[] radius, boolean sort) {
		return queryRadius(X.getData(), radius, sort);
	}
	
	public Neighborhood queryRadius(double[][] X, double[] radius, boolean sort) {
		int i, m_prime = X.length;
		int[] idx_arr_i, counts_arr;
		double[] dist_arr_i, pt;
		
		// Assumes non-jagged rows but caught in dist ops...
		MatUtils.checkDims(X);
		if(X[0].length != N_FEATURES)
			throw new DimensionMismatchException(X[0].length, N_FEATURES);
		
		VecUtils.checkDims(radius);
		if(m_prime != radius.length)
			throw new DimensionMismatchException(m_prime, radius.length);
		
		for(double rad: radius)
			ensurePositiveRadius(rad);
		
		// Prepare for iter
		int[][] indices = new int[m_prime][];
		double[][] dists= new double[m_prime][];
		
		idx_arr_i = new int[N_SAMPLES];
		dist_arr_i= new double[N_SAMPLES];
		counts_arr= new int[m_prime];
		
		
		// For each row in X
		for(i = 0; i < m_prime; i++) {
			// The current row
			pt = X[i];
			
			counts_arr[i] = queryRadiusSingle(0, pt, radius[i], 
											  idx_arr_i, 
											  dist_arr_i, 
											  0, true);
			
			if(sort)
				NeighborsHeap.simultaneous_sort(dist_arr_i, idx_arr_i, counts_arr[i]);
			
			
			// There's a chance the length could be zero if there are no neighbors in the radius...
			indices[i] = counts_arr.length == 0 ? new int[ ]{ }  : VecUtils.slice(idx_arr_i,  0, counts_arr[i]);
			dists[i]   = counts_arr.length == 0 ? new double[]{} : VecUtils.slice(dist_arr_i, 0, counts_arr[i]);
		}
		
		return new Neighborhood(dists, indices);
	}
	
	public Neighborhood queryRadius(double[][] X, double radius, boolean sort) {
		MatUtils.checkDims(X);
		ensurePositiveRadius(radius);
		
		int n = X[0].length;
		if(n != N_FEATURES)
			throw new DimensionMismatchException(n, N_FEATURES);
		
		return queryRadius(X, VecUtils.rep(radius, X.length), sort);
	}
	
	private int queryRadiusSingle(
			final int i_node, 
			final double[] pt, 
			final double r, 
			final int[] indices, 
			final double[] distances, 
			int count,
			final boolean returnDists) {
		
		double[][] data = this.data_arr;
		NodeData nodeInfo = node_data[i_node];
		
		int i;
		double reduced_r, dist_pt;
		
		// Lower bound (min)
		MutableDouble dist_LB = new MutableDouble(0.0);
				
		// Upper bound (max)
		MutableDouble dist_UB = new MutableDouble(0.0);
		
		// Find min dist and max dist from pts
		minMaxDist(this, i_node, pt, dist_LB, dist_UB);
		
		// If min dist is greater than radius, then pass
		if(dist_LB.value > r) {
		} // pass
		
		
		// All points within radius
		else if(dist_UB.value <= r) {
			for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
				/*// can't really happen?
				if(count < 0 || count >= N_SAMPLES) {
					String err = "count is too big; this should not happen";
					if(null != logger)
						logger.error(err);
					throw new IllegalStateException(err);
				}
				*/
				
				indices[count] = idx_array[i];
				if(returnDists)
					distances[count] = this.dist(pt, data[idx_array[i]]);
				
				count++;
			}
		}
		
		// this is a leaf node; check every point
		else if(nodeInfo.is_leaf) {
			reduced_r = this.dist_metric.distanceToPartialDistance(r);
			
			for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
				dist_pt = this.rDist(pt, data[idx_array[i]]);
				
				if(dist_pt <= reduced_r) {
					/*// can't really happen?
					if(count < 0 || count >= N_SAMPLES) {
						String err = "count is too big; this should not happen";
						if(null != logger)
							logger.error(err);
						throw new IllegalStateException(err);
					}
					*/
					
					indices[count] = idx_array[i];
					if(returnDists)
						distances[count] = this.dist_metric.partialDistanceToDistance(dist_pt);

					count++;
				}
			}
		}
		
		// Otherwise node is not a leaf. Recursively check subnodes
		else {
			count = this.queryRadiusSingle(2 * i_node + 1, pt, r, 
											indices, distances, count, 
											returnDists);
			
			count = this.queryRadiusSingle(2 * i_node + 2, pt, r, 
											indices, distances, count, 
											returnDists);
		}
		
		return count;
	}

	private void querySingleDepthFirst(int i_node, double[] pt, int i_pt, NeighborsHeap heap, double reduced_dist_LB) {
		NodeData nodeInfo = this.node_data[i_node];
		
		double dist_pt, reduced_dist_LB_1, reduced_dist_LB_2;
		int i, i1, i2;
		
		// Query point is outside node radius
		if(reduced_dist_LB > heap.largest(i_pt))
			this.n_trims++;
		
		// This is a leaf node
		else if(nodeInfo.is_leaf) {
			this.n_leaves++;
			for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
				dist_pt = rDist(pt, this.data_arr[idx_array[i]]);
				
				if(dist_pt < heap.largest(i_pt)) { // in radius
					heap.push(i_pt, dist_pt, idx_array[i]);
				}
			}
		}
		
		// Node is not a leaf
		else {
			this.n_splits++;
			i1 = 2 * i_node + 1;
			i2 = i1 + 1;
			
			reduced_dist_LB_1 = minRDist(this, i1, pt);
			reduced_dist_LB_2 = minRDist(this, i2, pt);
			
			// Recurse
			if(reduced_dist_LB_1 <= reduced_dist_LB_2) {
				querySingleDepthFirst(i1, pt, i_pt, heap, reduced_dist_LB_1);
				querySingleDepthFirst(i2, pt, i_pt, heap, reduced_dist_LB_2);
				
			} else { // opposite order
				
				querySingleDepthFirst(i2, pt, i_pt, heap, reduced_dist_LB_2);
				querySingleDepthFirst(i1, pt, i_pt, heap, reduced_dist_LB_1);
			}
		}
	}
	
	public int[] twoPointCorrelation(double[][] X, double r) {
		return twoPointCorrelation(X, r, false);
	}
	
	public int[] twoPointCorrelation(double[][] X, double r, boolean dual) {
		return twoPointCorrelation(X, VecUtils.rep(r, X.length), dual);
	}
	
	public int[] twoPointCorrelation(double[][] X, double[] r) {
		return twoPointCorrelation(X, r, false);
	}
	
	public int[] twoPointCorrelation(double[][] X, double[] r, boolean dual) {
		int i;
		
		MatUtils.checkDims(X);
		if(X[0].length != N_FEATURES)
			throw new DimensionMismatchException(X[0].length, N_FEATURES);
	
		double[][] Xarr = MatUtils.copy(X);
		double[] rarr = VecUtils.reorder(r, VecUtils.argSort(r));
		
		// count array
		int[] carr = new int[r.length];
		
		if(dual) {
			NearestNeighborHeapSearch other = newInstance(Xarr, leaf_size, dist_metric, logger);
			this.twoPointDual(0, other, 0, rarr, carr, 0, rarr.length);
		} else {
			for(i = 0; i < Xarr.length; i++)
				this.twoPointSingle(0, Xarr[i], rarr, carr, 0, rarr.length);
		}
		
		return carr;
	}
	
	private void twoPointDual(int i_node1, NearestNeighborHeapSearch other, int i_node2,
			double[] r, int[] count, int i_min, int i_max) {
		
		double[][] data1 = this.data_arr;
		double[][] data2 = other.data_arr;
		
		int[] idx_array1 = this.idx_array;
		int[] idx_array2 = other.idx_array;
		
		NodeData nodeInfo1 = this.node_data[i_node1];
		NodeData nodeInfo2 = other.node_data[i_node2];
		
		int i1, i2, j, Npts;
		double dist_pt;
		double dist_LB, dist_UB;
		
		dist_LB = minDistDual(this, i_node1, other, i_node2);
		dist_UB = maxDistDual(this, i_node1, other, i_node2);
		
		// Check for cuts
		while(i_min < i_max) {
			if(dist_LB > r[i_min])
				i_min++;
			else break;
		}
		
		while(i_max > i_min) {
			Npts = ((nodeInfo1.idx_end - nodeInfo1.idx_start) 
					* (nodeInfo2.idx_end - nodeInfo2.idx_start));
			if(dist_UB <= r[i_max - 1]) {
				count[i_max - 1] += Npts;
				i_max--;
			} else break;
		}
		
		if(i_min < i_max) {
			if(nodeInfo1.is_leaf && nodeInfo2.is_leaf) {
				for(i1 = nodeInfo1.idx_start; i1 < nodeInfo1.idx_end; i1++) {
					for(i2 = nodeInfo2.idx_start; i2 < nodeInfo2.idx_end; i2++) {
						
						dist_pt = this.dist(data1[idx_array1[i1]], data2[idx_array2[i2]]);
						j = i_max - 1;
						
						while(j >= i_min && dist_pt <= r[j])
							count[j--]++;
					}
				}
				
			} else if(nodeInfo1.is_leaf) {
				for(i2 = 2 * i_node2 + 1; i2 < 2 * i_node2 + 3; i2++)
					this.twoPointDual(i_node1, other, i2, r, count, i_min, i_max);
				
			} else if(nodeInfo2.is_leaf) {
				for(i1 = 2 * i_node1 + 1; i1 < 2 * i_node1 + 3; i1++)
					this.twoPointDual(i1, other, i_node2, r, count, i_min, i_max);
				
			} else {
				for(i1 = 2 * i_node1 + 1; i1 < 2 * i_node1 + 3; i1++)
					for(i2 = 2 * i_node2 + 1; i2 < 2 * i_node2 + 3; i2++)
						this.twoPointDual(i1, other, i2, r, count, i_min, i_max);
			}
		}
	}
	
	private void twoPointSingle(int i_node, double[] pt, double[] r, int[] count, int i_min, int i_max) {
		double[][] data = this.data_arr;
		NodeData nodeInfo = node_data[i_node];
		
		int i, j, Npts;
		double dist_pt;
		
		MutableDouble dist_LB = new MutableDouble(0.0), dist_UB = new MutableDouble(0.0);
		minMaxDist(this, i_node, pt, dist_LB, dist_UB);
		
		while(i_min < i_max) {
			if(dist_LB.value > r[i_min])
				i_min++;
			else break;
		}
		
		while(i_max > i_min) {
			Npts = nodeInfo.idx_end - nodeInfo.idx_start;
			if(dist_UB.value <= r[i_max - 1]) {
				count[i_max - 1] += Npts;
				i_max--;
			} else break;
			
		}
		
		if(i_min < i_max) {
			if(nodeInfo.is_leaf) {
				for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
					dist_pt = this.dist(pt, data[idx_array[i]]);
					j = i_max - 1;
					while(j >= i_min && dist_pt <= r[j])
						count[j--]++;
						// same as count[j]++; j--;
				}
			} else {
				this.twoPointSingle(2 * i_node + 1, pt, r, count, i_min, i_max);
				this.twoPointSingle(2 * i_node + 2, pt, r, count, i_min, i_max);
			}
		}
	}
	
	

	// Init functions
	abstract void allocateData	(NearestNeighborHeapSearch tree, int n_nodes, int n_features);
	abstract void initNode		(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end);
	
	// Dist functions
	//abstract double maxDist		(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double minDist		(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double maxDistDual	(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract double minDistDual	(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract void minMaxDist	(NearestNeighborHeapSearch tree, int i_node, double[] pt, MutableDouble minDist, MutableDouble maxDist);
	//abstract double maxRDist	(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double minRDist	(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract double minRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	
	// Hack for new instance functions
	abstract NearestNeighborHeapSearch newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger);
}