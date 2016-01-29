package com.clust4j.utils;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Loggable;
import com.clust4j.utils.NearestNeighborHeapSearch.Heap.NodeHeapData;

abstract public class NearestNeighborHeapSearch implements java.io.Serializable {
	private static final long serialVersionUID = -5617532034886067210L;
	
	final static public int DEF_LEAF_SIZE = 40;
	final static public DistanceMetric DEF_DIST = Distance.EUCLIDEAN;
	final static String MEM_ERR = "Internal: memory layout is flawed: " +
		"not enough nodes allocated";
	
	
	// Math constants for different kernels
	final static double LOG_PI  = FastMath.log(Math.PI);
	final static double LOG_2PI = FastMath.log(2 * Math.PI);
	final static double ROOT_2PI= FastMath.sqrt(2 * Math.PI);
	
	
	double[][] data_arr;
	int[] idx_array;
	NodeData[] node_data;
	double[][][] node_bounds;
	
	/** If there's a logger, for warnings will issue warn message */
	final Loggable logger;
	/** Constrained to Dist, not Sim due to nearest neighbor requirements */
	final DistanceMetric dist_metric;
	int n_trims, n_leaves, n_splits, n_calls, leaf_size, n_levels, n_nodes;
	
	
	
	
	
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X) {
		this(X, DEF_LEAF_SIZE, DEF_DIST);
	}
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, int leaf_size) {
		this(X, leaf_size, DEF_DIST);
	}
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, DistanceMetric dist) {
		this(X, DEF_LEAF_SIZE, dist);
	}
	
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, Loggable logger) {
		this(X, DEF_LEAF_SIZE, DEF_DIST, logger);
	}
	
	/**
	 * Default constructor without logger object
	 * @param X
	 * @param leaf_size
	 * @param dist
	 */
	public NearestNeighborHeapSearch(final AbstractRealMatrix X, int leaf_size, DistanceMetric dist) {
		this(X, leaf_size, dist, null);
	}
	
	/**
	 * Constructor with logger object
	 * @param X
	 * @param leaf_size
	 * @param dist
	 * @param logger
	 */
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
		
		// Add new NodeData objs to node_data arr
		this.node_data = new NodeData[n_nodes];
		for(int i = 0; i < node_data.length; i++)
			node_data[i] = new NodeData();
		
		// allocate tree specific data
		allocateData(this, n_nodes, n);
		recursiveBuild(0, 0, m);
	}
	
	
	
	
	
	// ========================== Inner classes ==========================
	/**
	 * Node data container
	 * @author Taylor G Smith
	 */
	static class NodeData {
		int idx_start, idx_end;
		boolean is_leaf;
		double radius;
		
		NodeData() { }
		NodeData(int st, int ed, boolean is, double rad) {
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
	}
	
	/**
	 * Abstract super class for NodeHeap and
	 * NeighborHeap classes
	 * @author Taylor G Smith
	 */
	abstract static class Heap {
		/** Node class. */
		static class NodeHeapData {
			double val;
			int i1;
			int i2;
			
			NodeHeapData() { }
			NodeHeapData(double val, int i1, int i2) {
				this.val = val;
				this.i1  = i1;
				this.i2  = i2;
			}
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
		
		EntryPair<double[][], int[][]> getArrays(boolean sort) {
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
			
			if(size <= 1){ // pass
			} else if(size == 2) {
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
	
	
	
	
	
	
	
	// ========================== Instance methods ==========================
	double dist(final double[] a, final double[] b) {
		return dist_metric.getDistance(a, b);
	}
	
	double rDist(final double[] a, final double[] b) {
		return dist_metric.getReducedDistance(a, b);
	}
	
	double rDistToDist(final double[] a, final double[] b) {
		return dist_metric.reducedDistanceToDistance(a, b);
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
	
	public static void partitionNodeIndices(double[][] data,
			int[] nodeIndices, int splitDim, int splitIndex,
			int nFeatures, int nPoints) {
			 
		int left = 0, right = nPoints - 1, midindex, i, j1, j2;
		double d1, d2;
		
		while(true) {
			midindex = left;
			
			for(i = left; i < right; i++) {
				j1 = nodeIndices[i] * nFeatures + splitDim;
				j2 = nodeIndices[right] * nFeatures + splitDim;
				
				d1 = data[j1 / nFeatures][j1 % nFeatures];
				d2 = data[j2 / nFeatures][j2 % nFeatures];
				            
				if(d1 < d2)
					swap(nodeIndices, i, midindex++);
			}
			
			swap(nodeIndices, midindex, right);
			if(midindex == splitIndex)
				break;
			
			left = (midindex < splitIndex) ? midindex + 1 : midindex - 1;
		}
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
	
	
	public int[][] query(double[][] X, int k, boolean dualTree, boolean breadthFirst, boolean sort) {
		MatUtils.checkDims(X);
		
		final int n = data_arr[0].length, m = data_arr.length;
		if(n != X[0].length)
			throw new DimensionMismatchException(n, X[0].length);
		if(m < k) throw new IllegalArgumentException(k+" is greater than rows in data");
		if(k < 1) throw new IllegalArgumentException(k+" must exceed 0");
		
		double[][] Xarr = MatUtils.copy(X);
		
		// Initialize neighbor heap
		NeighborsHeap heap = new NeighborsHeap(m, k);
		
		// Breadth-first node heap query
		NodeHeap nodeHeap = null;
		if(breadthFirst)
			nodeHeap = new NodeHeap(m / leaf_size);
		
		double[] bounds, pt;
		double reduced_dist_LB;
		n_trims  = 0;
		n_leaves = 0;
		n_splits = 0;
		
		if(dualTree) {
			NearestNeighborHeapSearch other = newInstance(Xarr, leaf_size, dist_metric, logger);
			
			if(breadthFirst)
				queryDualBreadthFirst(other, heap, nodeHeap);
			else {
				reduced_dist_LB = minRDistDual(this, 0, other, 0);
				bounds = VecUtils.rep(Double.POSITIVE_INFINITY, m);
				queryDualDepthFirst(0, other, 0, bounds, heap, reduced_dist_LB);
			}
			
		} else {
			int i;
			
			if(breadthFirst) {
				for(i = 0; i < X.length; i++) {
					pt = Xarr[i];
					querySingleBreadthFirst(pt, i, heap, nodeHeap);
				}
			} else {
				for(i = 0; i < X.length; i++) {
					pt = Xarr[i];
					reduced_dist_LB = minRDist(this, 0, pt);
					querySingleDepthFirst(0, pt, i, heap, reduced_dist_LB);
				}
			}
		}
		
		EntryPair<double[][], int[][]> distances_indices = heap.getArrays(sort);
		int[][] indices = distances_indices.getValue();
		
		
		return MatUtils.reshape(indices, X.length, k);
	}
	
	private void queryDualBreadthFirst(NearestNeighborHeapSearch other,
									   NeighborsHeap heap,
									   NodeHeap nodeHeap) {
		// dual-tree k-nn query breadth first
		int i1, i2, i_node1, i_node2, i_pt, 
			m = other.node_data.length;
		double dist_pt, reduced_dist_LB;
		double[] bounds = VecUtils.rep(Double.POSITIVE_INFINITY, m);
		
		NodeData[] node_data1 = this.node_data, 
			node_data2 = other.node_data;
		NodeData node_info1, node_info2;
		
		double[][] data1 = this.data_arr, data2 = other.data_arr;
		
		// Push nodes into node heap
		NodeHeapData nodeHeap_item = new NodeHeapData();
		nodeHeap_item.val = minRDistDual(this, 0, other, 0);
		nodeHeap_item.i1 = 0;
		nodeHeap_item.i2 = 0;
		nodeHeap.push(nodeHeap_item);
		
		
		while(nodeHeap.n > 0) {
			nodeHeap_item = nodeHeap.pop();
			reduced_dist_LB = nodeHeap_item.val;
			i_node1 = nodeHeap_item.i1;
			i_node2 = nodeHeap_item.i2;
			
			node_info1 = node_data1[i_node1];
			node_info2 = node_data2[i_node2];
			
			
			// If nodes are farther apart than current bound
			if(reduced_dist_LB > bounds[i_node2]) { // Pass here
			} 

			// If both nodes are leaves
			else if(node_info1.is_leaf && node_info2.is_leaf) {
				bounds[i_node2] = -1;
				
				
				for(i2 = node_info2.idx_start; i2 < node_info2.idx_end; i2++) {
					i_pt = other.idx_array[i2];
					
					if(heap.largest(i_pt) <= reduced_dist_LB)
						continue;
					
					
					for(i1 = node_info1.idx_start; i1 < node_info1.idx_end; i1++) {
						
						// sklearn line:
						// data1 + n_features * self.idx_array[i1],
                        // data2 + n_features * i_pt,
						dist_pt = rDist(data1[idx_array[i1]], data2[i_pt]);
						if(dist_pt < heap.largest(i_pt))
							heap.push(i_pt, dist_pt, idx_array[i1]);
					}
					
					// Keep track of node bound
					bounds[i_node2] = FastMath.max(bounds[i_node2], 
										heap.largest(i_pt));
				}
			}
			
			// When node 1 is a leaf or is smaller
			else if(node_info1.is_leaf 
					|| (!node_info2.is_leaf
						&& node_info2.radius > node_info1.radius)) {
				
				nodeHeap_item.i1 = i_node1;
				for(i2 = 2*i_node2+1; i2 < 2*i_node2+3; i2++) {
					nodeHeap_item.i2 = i2;
					nodeHeap_item.val = minRDistDual(this, i_node1, other, i2);
					nodeHeap.push(nodeHeap_item);
				}
			}
			
			// Otherwise node 2 is a leaf or is smaller
			else {
				nodeHeap_item.i2 = i_node2;
				for(i1 = 2*i_node1+1; i1 < 2*i_node1+3; i1++) {
					nodeHeap_item.i1 = i1;
					nodeHeap_item.val = minRDistDual(this, i1, other, i_node2);
					nodeHeap.push(nodeHeap_item);
				}
			}
		}
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
	
	private void querySingleBreadthFirst(double[] pt, int i_pt, NeighborsHeap heap, NodeHeap nodeHeap) {
		int i, i_node;
		double dist_pt, reduced_dist_LB;
		NodeData nodeInfo;
		
		NodeHeapData nodeHeap_item = new NodeHeapData();
		nodeHeap_item.val = minRDist(this, 0, pt);
		nodeHeap_item.i1 = 0;
		nodeHeap.push(nodeHeap_item);
		
		while(nodeHeap.n > 0) {
			nodeHeap_item = nodeHeap.pop();
			reduced_dist_LB = nodeHeap_item.val;
			i_node = nodeHeap_item.i1;
			nodeInfo = node_data[i_node];
			
			// Pt is outside radius:
			if(reduced_dist_LB < heap.largest(i_pt))
				this.n_trims++;
			
			// This is leaf node
			else if(nodeInfo.is_leaf) {
				this.n_leaves++;
				
				for(i = nodeInfo.idx_start; i < nodeInfo.idx_end; i++) {
					dist_pt = this.rDist(pt, this.data_arr[idx_array[i]]);
					if(dist_pt < heap.largest(i_pt))
						heap.push(i_pt, dist_pt, idx_array[i]);
				}
			}
			
			// Node is not a leaf
			else {
				this.n_splits++;
				for(i = 2 * i_node + 1; i < 2 * i_node + 3; i++) {
					nodeHeap_item.i1 = i;
					nodeHeap_item.val = minRDist(this, i, pt);
					nodeHeap.push(nodeHeap_item);
				}
			}
		}
	}
	
	private void querySingleDepthFirst(int i_node, double[] pt, int i_pt, NeighborsHeap heap, double reduced_dist_LB) {
		NodeData nodeInfo = node_data[i_node];
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
				if(dist_pt < heap.largest(i_pt)) // in radius
					heap.push(i_pt, dist_pt, idx_array[i]);
			}
		}
		
		// Node is not a leaf
		else {
			this.n_splits++;
			i1 = 2 * i_node + 1;
			i2 = i1 + 1;
			
			reduced_dist_LB_1 = minRDist(this, i1, pt);
			reduced_dist_LB_2 = minRDist(this, i2, pt);
			
			if(reduced_dist_LB_1 <= reduced_dist_LB_2) {
				querySingleDepthFirst(i1, pt, i_pt, heap, reduced_dist_LB_1);
				querySingleDepthFirst(i2, pt, i_pt, heap, reduced_dist_LB_2);
				
			} else { // opposite order
				
				querySingleDepthFirst(i2, pt, i_pt, heap, reduced_dist_LB_2);
				querySingleDepthFirst(i1, pt, i_pt, heap, reduced_dist_LB_1);
			}
		}
	}
	
	

	abstract void allocateData(NearestNeighborHeapSearch tree, int n_nodes, int n_features);
	abstract void initNode(NearestNeighborHeapSearch tree, int i_node, int idx_start, int idx_end);
	abstract double maxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double minDist(NearestNeighborHeapSearch tree, int i_node, double[] pt);
	abstract double maxDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract double minDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract double minMaxDist(NearestNeighborHeapSearch tree, int i_node, double[] pt, double lb, double ub);
	abstract double maxRDist(NearestNeighborHeapSearch tree, int i_node, double[] a);
	abstract double minRDist(NearestNeighborHeapSearch tree, int i_node, double[] a);
	abstract double maxRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract double minRDistDual(NearestNeighborHeapSearch tree1, int iNode1, NearestNeighborHeapSearch tree2, int iNode2);
	abstract NearestNeighborHeapSearch newInstance(double[][] arr, int leaf, DistanceMetric dist);
	abstract NearestNeighborHeapSearch newInstance(double[][] arr, int leaf, DistanceMetric dist, Loggable logger);
}