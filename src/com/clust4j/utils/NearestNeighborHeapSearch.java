package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;

abstract public class NearestNeighborHeapSearch implements java.io.Serializable {
	private static final long serialVersionUID = -5617532034886067210L;
	
	final static public DistanceMetric DEF_DIST = Distance.EUCLIDEAN;
	final static String MEM_ERR = "Internal: memory layout is flawed: " +
		"not enough nodes allocated";
	
	double[][] data_arr;
	int[] idx_array;
	NodeData[] node_data;
	double[][][] node_bounds;
	
	final Loggable logger;
	final DistanceMetric dist_metric;
	int n_trims, n_leaves, n_splits, n_calls, leaf_size, n_levels, n_nodes;
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		this(X, leaf_size, dist, null);
	}
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist, Loggable logger) {
		this.data_arr = X.getData();
		this.leaf_size = leaf_size;
		this.dist_metric = dist;
		this.logger = logger;
		
		if(leaf_size < 1)
			throw new IllegalArgumentException("illegal leaf size: " + leaf_size);
		
		
		// determine number of levels in the tree, and from this
        // the number of nodes in the tree.  This results in leaf nodes
        // with numbers of points between leaf_size and 2 * leaf_size
		int m = data_arr.length, n = X.getColumnDimension();
		this.n_levels = (int)FastMath.log(2, FastMath.max(1, (m-1)/leaf_size)) + 1;
		this.n_nodes = (int)(FastMath.pow(2, n_levels)) - 1;
	
		// allocate arrays for storage
		this.idx_array = VecUtils.arange(m);
		this.node_data = new NodeData[n_nodes];
		
		// allocate tree specific data
		allocateData(this, n_nodes, n);
		recursiveBuild(0, 0, m);
	}
	
	
	
	/**
	 * Node data container
	 * @author Taylor G Smith
	 */
	static class NodeData {
		int idx_start, idx_end;
		boolean is_leaf;
		double radius;
		
		NodeData(int st, int ed, boolean is, double rad) {
			idx_start = st;
			idx_end = ed;
			is_leaf = is;
			radius = rad;
		}
	}
	
	/**
	 * Abstract super class for NodeHeap and
	 * NeighborHeap classes
	 * @author Taylor G Smith
	 */
	abstract static class Heap {
		
		/**
		 * Node class.
		 * @author Taylor G Smith
		 */
		static class NodeHeapData {
			double val;
			int i1;
			int i2;
		}
		
		Heap(){}
		
		static void swapNodes(NodeHeapData[] arr, int i1, int i2) {
			NodeHeapData tmp = arr[i1];
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
		double[][] distances;
		int[][] indices;
		
		NeighborsHeap(int nPts, int k) {
			super();
			distances = MatUtils.rep(Double.POSITIVE_INFINITY, nPts, k);
			indices   = new int[nPts][k];
		}
		
		EntryPair<double[][], int[][]> get_arrays(boolean sort) {
			if(sort)
				sort();
			return new EntryPair<>(distances, indices);
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
			for(int row = 0; row < distances.length; row++)
				simultaneous_sort(distances[row], indices[row], distances[row].length);
			
			return 0;
		}
		
		double largest(int row) {
			return distances[row][0];
		}
		
		int simultaneous_sort(double[] dist, int[] idx, int size) {
			int pivot_idx, i, store_idx;
			double pivot_val;
			
			if(size <= 1)
				return 0;
			else if(size == 2) {
				if(dist[0] > dist[1])
					dualSwap(dist, idx, 0, 1);
			} else if(size == 3) {
				if(dist[0] > dist[1])
					dualSwap(dist, idx, 0, 1);
				if(dist[1] > dist[2]) {
					dualSwap(dist, idx, 1, 2);
					if(dist[0] > dist[1])
						dualSwap(dist, idx, 0, 1);
				}
			} else {
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
				if(pivot_idx + 2 < size)
					simultaneous_sort(
						VecUtils.slice(dist, pivot_idx+1, dist.length),
						VecUtils.slice(idx,  pivot_idx+1, idx.length),
						size - pivot_idx - 1);
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
		NodeHeapData[] data;
		int n;
		
		NodeHeap(int size) {
			super();
			size = FastMath.max(size, 1);
			data = new NodeHeapData[size];
			n = size;
			clear();
		}
		
		void clear() {
			n = 0;
		}
		
		NodeHeapData peek() {
			return data[0];
		}
		
		NodeHeapData pop() {
			if(n == 0)
				throw new IllegalStateException("cannot pop an empty heap");
			
			int i, i_child1, i_child2, i_swap;
			NodeHeapData popped_element = data[0];
			
			// pop off the first element, move the last element to the front,
	        // and then perform swaps until the heap is back in order
			data[0] = data[n - 1];
			n--;
			
			i = 0;
			while(i < n) {
				i_child1 = 2 * i + 1;
				i_child2 = 2 * i + 2;
				i_swap = 0;
				
				if(i_child2 < n) {
					if(data[i_child1].val <= data[i_child2].val)
						i_swap = i_child1;
					else
						i_swap = i_child2;
				} else if(i_child1 < n)
					i_swap = i_child1;
				else
					break;
				
				if(i_swap > 0 && data[i_swap].val <= data[i].val) {
					swapNodes(data, i, i_swap);
					i = i_swap;
				} else
					break;
			}
			
			return popped_element;
		}
		
		int push(NodeHeapData node) {
			// Add to the heap
			int i = n, i_parent;
			n++;
			
			if(n > this.data.length)
				resize(2 * n);
			
			// Put new element at end, perform swaps
			this.data[i] = node;
			
			while(i > 0) {
				i_parent = (i-1) / 2;
				if(data[i_parent].val <= data[i].val)
					break;
				else {
					swapNodes(data, i, i_parent);
					i = i_parent;
				}
			}
			
			return 0;
		}
		
		int resize(int new_size) {
			// Resize larger or smaller
			int size = data.length, lim = FastMath.min(size, new_size);
			NodeHeapData[] newData = new NodeHeapData[new_size];
			
			if(size > 0 && new_size > 0)
				for(int i = 0; i < lim; i++)
					newData[i] = data[i];
			
			if(new_size < size)
				n = new_size;
			
			this.data = newData;
			return 0;
		}
	}
	
	
	
	
	
	double dist(final double[] a, final double[] b) {
		return dist_metric.getDistance(a, b);
	}
	
	// Tested: passing
	public static int findNodeSplitDim(double[][] data, int[] idcs) {
		// Gets the difference between the vector of column
		// maxes and the vector of column mins, then finds the
		// arg max.
		
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
	
	public int getNumCalls() {
		return n_calls;
	}
	
	public TriTup<Integer, Integer, Integer> getTreeStats() {
		return new TriTup<>(n_trims, n_leaves, n_splits);
	}
	
	public static int partitionNodeIndices(double[][] data, int[] idcs,
			int split_dim, int split_index, int n_features, int n_points) {
		
		int left = 0, right = n_points - 1, midIndex, i;
		double[] d1, d2;
		
		while(true) {
			midIndex = left;
			
			
			for(int idx: idcs) {
				for(i = left; i < right; i++) {
					
					
					d1 = data[idcs[i] * n_features + split_dim];
					d2 = data[idcs[right] * n_features + split_dim];
					
					// TODO
				}
			}
			
			swap(idcs, midIndex, right);
			if(midIndex == split_index)
				break;
			else if(midIndex < split_index)
				left = midIndex + 1;
			else
				right= midIndex - 1;
		}
		
		return 0;
	}
	
	void resetNumCalls() {
		n_calls = 0;
	}
	
	void recursiveBuild(int i_node, int idx_start, int idx_end) {
		int i_max, n = data_arr[0].length, n_points = idx_end - idx_start,
			n_mid = n_points / 2;
		initNode(this, i_node, idx_start, idx_end);
		
		
		if(2 * i_node + 1 >= n_nodes) {
			node_data[i_node].is_leaf = true;
			if(idx_end - idx_start > 2 * leaf_size && null != logger)
				logger.warn(MEM_ERR);
			
		} else if(idx_end - idx_start < 2) {
			if(null != logger)
				logger.warn(MEM_ERR);
			node_data[i_node].is_leaf = true;
		} else {
			// split node and recursively build child nodes
			node_data[i_node].is_leaf = false;
			i_max = findNodeSplitDim(data_arr, idx_array);
			partitionNodeIndices(data_arr, idx_array, 
					i_max, n_mid, n, n_points);
			
			recursiveBuild(2 * i_node + 1, idx_start, idx_start + n_mid);
			recursiveBuild(2 * i_node + 2, idx_start + n_mid, idx_end);
		}
	}
	
	static void swap(int[] idcs, int i1, int i2) {
		int tmp = idcs[i1];
		idcs[i1] = idcs[i2];
		idcs[i2] = tmp;
	}

	public abstract void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features);
	public abstract void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end);
}