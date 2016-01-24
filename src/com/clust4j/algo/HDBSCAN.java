package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.algo.HDBSCAN.HList;
import com.clust4j.utils.QuadTup;
import com.clust4j.utils.TriTup;
import com.clust4j.algo.HierarchicalAgglomerative.AverageLinkageTree;
import com.clust4j.algo.HierarchicalAgglomerative.CompleteLinkageTree;
import com.clust4j.algo.HierarchicalAgglomerative.HierarchicalDendrogram;
import com.clust4j.algo.HierarchicalAgglomerative.WardTree;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.Distance;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.HaversineDistance;
import com.clust4j.utils.Inequality;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MinkowskiDistance;
import com.clust4j.utils.MatUtils.MatSeries;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.VecUtils.VecSeries;

/**
 * Hierarchical Density-Based Spatial Clustering of Applications with Noise. 
 * Performs {@link DBSCAN} over varying epsilon values and integrates the result to 
 * find a clustering that gives the best stability over epsilon. This allows 
 * HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more 
 * robust to parameter selection.
 * 
 * @author Taylor G Smith, adapted from the Python 
 * <a href="https://github.com/lmcinnes/hdbscan">HDBSCAN package</a>, inspired by
 * <a href="http://dl.acm.org/citation.cfm?id=2733381">the paper</a> by 
 * R. Campello, D. Moulavi, and J. Sander
 */
public class HDBSCAN extends AbstractDBSCAN {
	private static final long serialVersionUID = -5112901322434131541L;
	public static final Algorithm DEF_ALGO = Algorithm.GENERIC;
	public static final double DEF_ALPHA = 1.0;
	public static final boolean DEF_APPROX_MIN_SPAN = true;
	public static final int DEF_LEAF_SIZE = 40;
	public static final int DEF_MIN_CLUST_SIZE = 5;
	//public static final boolean DEF_GENERATE_MIN_SPAN = false;
	
	private final Algorithm algo;
	private final double alpha;
	private final boolean approxMinSpanTree;
	private final int min_cluster_size;
	private final int leafSize;
	//private final boolean genMinSpanTree;

	private volatile HDBSCANLinkageTree tree = null;
	private volatile double[][] dist_mat = null;
	private volatile int[] labels = null;
	private volatile int numClusters;
	private volatile int numNoisey;
	
	
	public static enum Algorithm {
		GENERIC,
		PRIMS_KD_TREE,
		PRIMS_BALLTREE,
		BORUVKA_KDTREE,
		BORUVKA_BALLTREE
	}
	
	
	
	public final static ArrayList<Class<? extends GeometricallySeparable>> ValidKDMetrics;
	static {
		ValidKDMetrics = new ArrayList<>();
		ValidKDMetrics.add(Distance.EUCLIDEAN.getClass());
		ValidKDMetrics.add(Distance.MANHATTAN.getClass());
		ValidKDMetrics.add(MinkowskiDistance.class);
		ValidKDMetrics.add(Distance.CHEBYSHEV.getClass());
	}
	
	
	/**
	 * Constructs an instance of HDBSCAN from the default values
	 * @param data
	 */
	public HDBSCAN(final AbstractRealMatrix data) {
		this(data, DEF_MIN_PTS);
	}
	
	/**
	 * Constructs an instance of HDBSCAN from the default values
	 * @param eps
	 * @param data
	 */
	public HDBSCAN(final AbstractRealMatrix data, final int minPts) {
		this(data, new HDBSCANPlanner(minPts));
	}
	
	/**
	 * Constructs an instance of HDBSCAN from the provided builder
	 * @param builder
	 * @param data
	 */
	public HDBSCAN(final AbstractRealMatrix data, final HDBSCANPlanner planner) {
		super(data, planner);
		this.algo = planner.algo;
		this.alpha = planner.alpha;
		this.approxMinSpanTree = planner.approxMinSpanTree;
		this.min_cluster_size = planner.min_cluster_size;
		this.leafSize = planner.leafSize;
		//this.genMinSpanTree = planner.genMinTree;
		
		if(alpha == 0.0)
			throw new ArithmeticException("alpha cannot equal 0");
		
		meta("min_cluster_size="+min_cluster_size);
		meta("min_pts="+planner.minPts);
		meta("algorithm="+algo);
		meta("alpha="+alpha);
	}
	
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for HDBSCAN
	 * @author Taylor G Smith
	 */
	final public static class HDBSCANPlanner extends AbstractDBSCANPlanner {
		private int minPts = DEF_MIN_PTS;
		private boolean scale = DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private Random seed = DEF_SEED;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private Algorithm algo = DEF_ALGO;
		private double alpha = DEF_ALPHA;
		private boolean approxMinSpanTree = DEF_APPROX_MIN_SPAN;
		private int min_cluster_size = DEF_MIN_CLUST_SIZE;
		private int leafSize = DEF_LEAF_SIZE;
		//private boolean genMinTree = DEF_GENERATE_MIN_SPAN;
		
		
		public HDBSCANPlanner() { }
		public HDBSCANPlanner(final int minPts) {
			this.minPts = minPts;
		}

		
		@Override
		public HDBSCAN buildNewModelInstance(AbstractRealMatrix data) {
			return new HDBSCAN(data, this);
		}
		
		@Override
		public HDBSCANPlanner copy() {
			return new HDBSCANPlanner(minPts)
				.setAlgo(algo)
				.setAlpha(alpha)
				.setApprox(approxMinSpanTree)
				.setLeafSize(leafSize)
				.setMinClustSize(min_cluster_size)
				//.setGenMinSpan(genMinTree)
				.setScale(scale)
				.setSep(dist)
				.setSeed(seed)
				.setVerbose(verbose)
				.setNormalizer(norm);
		}

		@Override
		public int getMinPts() {
			return minPts;
		}
		
		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public HDBSCANPlanner setAlgo(final Algorithm algo) {
			this.algo = algo;
			return this;
		}
		
		public HDBSCANPlanner setAlpha(final double a) {
			this.alpha = a;
			return this;
		}
		
		public HDBSCANPlanner setApprox(final boolean b) {
			this.approxMinSpanTree = b;
			return this;
		}
		
		public HDBSCANPlanner setLeafSize(final int leafSize) {
			this.leafSize = leafSize;
			return this;
		}
		
		public HDBSCANPlanner setMinClustSize(final int min) {
			this.min_cluster_size = min;
			return this;
		}
		
		/*public HDBSCANPlanner setGenMinSpan(final boolean b) {
			this.genMinTree = b;
			return this;
		}*/
		
		@Override
		public HDBSCANPlanner setMinPts(final int minPts) {
			this.minPts = minPts;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public HDBSCANPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public HDBSCANPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
		
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}
		
		@Override
		public HDBSCANPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
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
		
		NeighborsHeap() {
			this(true);
		}
		
		NeighborsHeap(boolean sort) {
			//TODO
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
	
	abstract static class BinaryTree implements java.io.Serializable {
		private static final long serialVersionUID = -5617532034886067210L;
		final static String MEM_ERR = "Internal: memory layout is flawed: " +
			"not enough nodes allocated";
		
		public final static ArrayList<Class<? extends GeometricallySeparable>> VALID_METRICS;
		static {
			VALID_METRICS = new ArrayList<>();
			VALID_METRICS.add(Distance.BRAY_CURTIS.getClass());
			VALID_METRICS.add(Distance.CANBERRA.getClass());
			VALID_METRICS.add(Distance.DICE.getClass());
			VALID_METRICS.add(Distance.HAMMING.getClass());
			VALID_METRICS.add(HaversineDistance.class);
			VALID_METRICS.add(Distance.KULSINSKI.getClass());
			VALID_METRICS.add(Distance.ROGERS_TANIMOTO.getClass());
			VALID_METRICS.add(Distance.RUSSELL_RAO.getClass());
			VALID_METRICS.add(Distance.SOKAL_SNEATH.getClass());
			VALID_METRICS.addAll(ValidKDMetrics);
			
		}
		
		double[][] data_arr;
		int[] idx_array;
		NodeData[] node_data;
		double[][][] node_bounds;
		final HDBSCAN model;

		GeometricallySeparable dist_metric;
		int n_trims, n_leaves, n_splits, n_calls, leaf_size, n_levels, n_nodes;
		
		BinaryTree(final AbstractRealMatrix X, int leaf_size, HDBSCAN model) {
			this.data_arr = X.getData();
			this.leaf_size = leaf_size;
			
			
			String err;
			if(leaf_size < 1) {
				err = "illegal leaf size: " + leaf_size;
				model.error(err);
				throw new IllegalArgumentException(err);
			}
			
			Class<? extends GeometricallySeparable> clz = 
				model.getSeparabilityMetric().getClass();
			if( !ValidKDMetrics.contains(clz) ) {
				model.warn(clz+" is not a valid distance metric for BinaryTrees. Valid metrics: " + VALID_METRICS);
				model.warn("falling back to default metric: " + AbstractClusterer.DEF_DIST);
				model.setSeparabilityMetric(AbstractClusterer.DEF_DIST);
			}
			
			this.model = model;
			this.dist_metric = model.getSeparabilityMetric();
			
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
		
		double dist(final double[] a, final double[] b) {
			return dist_metric.getDistance(a, b);
		}
		
		// Tested: passing
		static int findNodeSplitDim(double[][] data, int[] idcs) {
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
		
		QuadTup<double[][], int[], NodeData[], double[][][]> getArrays() {
			return new QuadTup<>(data_arr, idx_array, node_data, node_bounds);
		}
		
		int get_n_calls() {
			return n_calls;
		}
		
		TriTup<Integer, Integer, Integer> getTreeStats() {
			return new TriTup<>(n_trims, n_leaves, n_splits);
		}
		
		static int partitionNodeIndices(double[][] data, int[] idcs,
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
		
		void reset_n_calls() {
			n_calls = 0;
		}
		
		void recursiveBuild(int i_node, int idx_start, int idx_end) {
			int i_max, n = data_arr[0].length, n_points = idx_end - idx_start,
				n_mid = n_points / 2;
			initNode(this, i_node, idx_start, idx_end);
			
			
			if(2 * i_node + 1 >= n_nodes) {
				node_data[i_node].is_leaf = true;
				if(idx_end - idx_start > 2 * leaf_size)
					model.warn(MEM_ERR);
				
			} else if(idx_end - idx_start < 2) {
				model.warn(MEM_ERR);
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

		abstract void allocateData(BinaryTree tree, int n_nodes, int n_features);
		abstract void initNode(BinaryTree tree, int i_node, int idx_start, int idx_end);
	}
	
	
	static class BinaryKDTree extends BinaryTree {
		private static final long serialVersionUID = -8176392242423193895L;
		
		public BinaryKDTree(final AbstractRealMatrix X, int leaf_size, HDBSCAN model) {
			super(X, leaf_size, model);
		}

		@Override
		void allocateData(BinaryTree tree, int n_nodes, int n_features) {
			tree.node_bounds = new double[2][n_nodes][n_features];
		}

		@Override
		void initNode(BinaryTree tree, int i_node, int idx_start, int idx_end) {
			// TODO
			
		}
		
		// TODO
	}
	
	static class BinaryBallTree extends BinaryTree {
		private static final long serialVersionUID = 3745191326510760880L;
		
		public BinaryBallTree(final AbstractRealMatrix X, int leaf_size, HDBSCAN model) {
			super(X, leaf_size, model);
		}

		@Override
		void allocateData(BinaryTree tree, int n_nodes, int n_features) {
			tree.node_bounds = new double[1][n_nodes][n_features];
		}

		@Override
		void initNode(BinaryTree tree, int i_node, int idx_start, int idx_end) {
			// TODO Auto-generated method stub
			
		}
		
		// TODO
	}
	
	final static class HList<T> extends ArrayList<T> {
		private static final long serialVersionUID = 2784009809720305029L;
		
		public HList() {
			super();
		}
		
		public HList(Collection<? extends T> coll) {
			super(coll);
		}
		
		public HList(T[] t) {
			super(t.length);
			for(T tee: t)
				add(tee);
		}
		
		public T pop() {
			if(this.isEmpty())
				return null;
			
			final T t = this.get(this.size() - 1);
			this.remove(this.size() - 1);
			return t;
		}
		
		public void push(T t) {
			if(this.isEmpty())
				add(t);
			else
				add(0, t);
		}
	}
	
	interface Boruvka {}
	interface Prim { 
		double[][] mstLinkageCore(final double[][] X, final double[][] coreDists); 
	}
	
	/**
	 * Util mst linkage methods
	 * @author Taylor G Smith
	 */
	static class LinkageTreeUtils {	
		
		/**
		 * Perform a breadth first search on a tree
		 * @param hierarchy
		 * @param root
		 * @return
		 */
		// Tested: passing
		static HList<Integer> breadthFirstSearch(final double[][] hierarchy, final int root) {
			HList<Integer> toProcess = new HList<>(), tmp;
			int dim = hierarchy.length, maxNode = 2*dim, numPoints = maxNode - dim+1;
			
			toProcess.add(root);
			HList<Integer> result = new HList<>();
			while(!toProcess.isEmpty()) {
				result.addAll(toProcess);
				
				tmp = new HList<>();
				for(Integer x: toProcess)
					if(x >= numPoints)
						tmp.add(x - numPoints);
				toProcess = tmp;
				
				tmp = new HList<>();
				if(!toProcess.isEmpty()) {
					for(Integer row: toProcess)
						for(int i = 0; i < 2; i++)
							tmp.add((int) hierarchy[row][wraparoundIdxGet(hierarchy[row].length, i)]);
					
					toProcess = tmp;
				}
			}
			
			return result;
		}
		
		// Tested: passing
		static TreeMap<Integer, Double> computeStability(HList<QuadTup<Integer, Integer, Double, Integer>> condensed) {
			double[] resultArr, births, lambdas = new double[condensed.size()];
			int[] sizes = new int[condensed.size()], parents = new int[condensed.size()];
			int child, parent, childSize, resultIdx, currentChild = -1, idx = 0, row = 0;
			double lambda, minLambda = 0;
			
			
			
			// ['parent', 'child', 'lambda', 'childSize']
			// Calculate starting maxes/mins
			int largestChild = Integer.MIN_VALUE,
				minParent = Integer.MAX_VALUE,
				maxParent = Integer.MIN_VALUE;
			
			int[] sortedChildren= new int[condensed.size()];
			double[] sortedLambdas = new double[condensed.size()];
			
			for(QuadTup<Integer, Integer, Double, Integer> q: condensed) {
				parent= q.one;
				child = q.two;
				lambda= q.three;
				childSize= q.four;
				
				if(child > largestChild)
					largestChild = child;
				if(parent < minParent)
					minParent = parent;
				if(parent > maxParent)
					maxParent = parent;
				
				parents[idx] = parent;
				sizes[idx]= childSize;
				lambdas[idx]= lambda;
				
				sortedChildren[idx] = child;
				sortedLambdas[idx++]= lambda;
			}
			
			int numClusters = maxParent - minParent + 1;
			births = VecUtils.rep(Double.NaN, largestChild + 1);
			Arrays.sort(sortedChildren);
			Arrays.sort(sortedLambdas);
			
			// Start first loop
			for(row = 0; row < sortedChildren.length; row++) {
				child = sortedChildren[row]; // 0,1,2 in test
				lambda= sortedLambdas[row];  // 1.667 in test
				
				if(child == currentChild)
					minLambda = FastMath.min(minLambda, lambda);
				else if(currentChild != -1) {
					// Already been initialized
					births[currentChild] = minLambda;
					currentChild = child;
					minLambda = lambda;
				} else {
					// Initialize
					currentChild = child;
					minLambda = lambda;
				}
			}

			resultArr = new double[numClusters];
			
			
			// Second loop
			double birthParent;
			for(idx = 0; idx < condensed.size(); idx++) {
				parent = parents[idx];
				lambda = lambdas[idx];
				childSize= sizes[idx];
				resultIdx = parent - minParent;
				
				// the Cython exploits the C contiguous pointer array's
				// out of bounds allowance (2.12325E-314), but we have to
				// do a check for that...
				birthParent = parent >= births.length ? GlobalState.Mathematics.TINY : births[parent];
				resultArr[resultIdx] += (lambda - birthParent) * childSize;
			}
			
			
			double[] top = VecUtils.asDouble(VecUtils.arange(minParent, maxParent + 1));
			double[][] mat = MatUtils.transpose(VecUtils.vstack(top, resultArr));
			
			TreeMap<Integer, Double> result = new TreeMap<>();
			for(idx = 0; idx < mat.length; idx++)
				result.put( (int)mat[idx][0], mat[idx][1]);
			
			return result;
		}
		
		// Tested: passing
		static HList<QuadTup<Integer, Integer, Double, Integer>> condenseTree(final double[][] hierarchy, final int minSize) {
			final int m = hierarchy.length;
			int root = 2 * m, numPoints = root/2 + 1 /*Integer division*/, nextLabel = numPoints+1;
			HList<Integer> nodeList = breadthFirstSearch(hierarchy, root), tmpList;
			HList<QuadTup<Integer, Integer, Double, Integer>> resultList = new HList<>();
			int[] relabel = new int[nodeList.size()]; 
			boolean[] ignore = new boolean[nodeList.size()];
			double[] children;
			
			double lambda;
			int left, right, leftCount, rightCount;
			relabel[root] = numPoints;
			
			
			
			for(Integer node: nodeList) {
				
				if(ignore[node] || node < numPoints)
					continue;
				
				children = hierarchy[wraparoundIdxGet(hierarchy.length, node-numPoints)];
				left = (int) children[0];
				right= (int) children[1];
				
				if(children[2] > 0)
					lambda = 1.0 / children[2];
				else lambda = Double.POSITIVE_INFINITY;
				
				if(left >= numPoints)
					leftCount = (int) (hierarchy[wraparoundIdxGet(hierarchy.length, left-numPoints)][3]);
				else leftCount = 1;
				
				if(right >= numPoints)
					rightCount = (int)(hierarchy[wraparoundIdxGet(hierarchy.length,right-numPoints)][3]);
				else rightCount = 1;
				
				
				
				if(leftCount >= minSize && rightCount >= minSize) {
					relabel[left] = nextLabel++;
					resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
						relabel[wraparoundIdxGet(relabel.length, node)],
						relabel[wraparoundIdxGet(relabel.length, left)],
						lambda, leftCount ));
					
					relabel[wraparoundIdxGet(relabel.length, right)] = nextLabel++;
					resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
						relabel[wraparoundIdxGet(relabel.length, node)],
						relabel[wraparoundIdxGet(relabel.length,right)],
						lambda, rightCount ));
					
					
				} else if(leftCount < minSize && rightCount < minSize) {
					tmpList = breadthFirstSearch(hierarchy, left);
					for(Integer subnode: tmpList) {
						if(subnode < numPoints)
							resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
								relabel[wraparoundIdxGet(relabel.length, node)], subnode,
								lambda, 1));
						ignore[subnode] = true;
					}
					
					tmpList = breadthFirstSearch(hierarchy, right);
					for(Integer subnode: tmpList) {
						if(subnode < numPoints)
							resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
								relabel[wraparoundIdxGet(relabel.length, node)], subnode,
								lambda, 1));
						ignore[subnode] = true;
					}
					
					
 				} else if(leftCount < minSize) {
 					relabel[right] = relabel[node];
 					tmpList = breadthFirstSearch(hierarchy, left);
 					
 					for(Integer subnode: tmpList) {
						if(subnode < numPoints)
							resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
								relabel[wraparoundIdxGet(relabel.length, node)], subnode,
								lambda, 1));
						ignore[subnode] = true;
					}
 				}
				
				
 				else {
 					relabel[left] = relabel[node];
 					tmpList = breadthFirstSearch(hierarchy, right);
 					for(Integer subnode: tmpList) {
						if(subnode < numPoints)
							resultList.add(new QuadTup<Integer, Integer, Double, Integer>(
								relabel[wraparoundIdxGet(relabel.length, node)], subnode,
								lambda, 1));
						ignore[subnode] = true;
					}
 				}
			}
			
			return resultList;
		}
		
		/**
		 * Generic linkage core method
		 * @param X
		 * @param m
		 * @return
		 */
		static double[][] mstLinkageCore(final double[][] X, final int m) { // Tested: passing
			int[] node_labels, current_labels, tmp_labels; 
			double[] current_distances, left, right;
			boolean[] label_filter;
			boolean val;
			int current_node, new_node_index, new_node, i, j, trueCt, idx;
			VecSeries series;
			
			double[][] result = new double[m-1][3];
			node_labels = VecUtils.arange(m);
			current_node = 0;
			current_distances = VecUtils.rep(Double.POSITIVE_INFINITY, m);
			current_labels = node_labels;
			
			
			
			for(i = 1; i < node_labels.length; i++) {
				
				// Create the boolean mask; takes 2N to create mask and then filter
				// however, creating the left vector concurrently 
				// trims off one N pass. This could be done using Vector.VecSeries
				// but that would add an extra pass of N
				idx = 0;
				trueCt = 0;
				label_filter = new boolean[current_labels.length];
				for(j = 0; j < label_filter.length; j++) {
					val = current_labels[j] != current_node;
					if(val)
						trueCt++;
					
					label_filter[j] = val;
				}
				
				tmp_labels = new int[trueCt];
				left = new double[trueCt];
				for(j = 0; j < current_labels.length; j++) {
					if(label_filter[j]) {
						tmp_labels[idx] = current_labels[j];
						left[idx] = current_distances[j];
						idx++;
					}
				}
				
				current_labels = tmp_labels;
				right = new double[current_labels.length];
				for(j = 0; j < right.length; j++)
					right[j] = X[current_node][current_labels[j]];
				
				// Build the current_distances vector
				series = new VecSeries(left, Inequality.LT, right);
				current_distances = VecUtils.where(series, left, right);
				
				
				// Get next iter values
				new_node_index = VecUtils.argMin(current_distances);
				new_node = current_labels[new_node_index];
				result[i-1][0] = (double)current_node;
				result[i-1][1] = (double)new_node;
				result[i-1][2] = current_distances[new_node_index];
				
				current_node = new_node;
			}
			
			return result;
		}
		
		static double[][] mstLinkageCore_cdist(final double[][] X, final int m) {
			// TODO:
			return null;
		}
		

		
		/**
		 * The index may be -1; this will return 
		 * the index of the length of the array minus
		 * the absolute value of the index in the case
		 * of negative indices, like the original Python
		 * code.
		 * @param array
		 * @param idx
		 * @throws ArrayIndexOutOfBoundsException if the absolute value of the index
		 * exceeds the length of the array
		 * @return the index to be queried in wrap-around indexing
		 */
		static int wraparoundIdxGet(int array_len, int idx) {
			int abs;
			if((abs = FastMath.abs(idx)) > array_len)
				throw new ArrayIndexOutOfBoundsException(idx);
			if(idx >= 0)
				return idx;
			return array_len - abs;
		}
	}
	
	abstract class HDBSCANLinkageTree {
		final HDBSCAN model;
		final GeometricallySeparable metric;
		final int m, n;
		
		HDBSCANLinkageTree() {
			model = HDBSCAN.this;
			metric = model.getSeparabilityMetric();
			m = model.data.getRowDimension();
			n = model.data.getColumnDimension();
			
			// Can only happen if this class is instantiated
			// from an already-trained HDBSCAN instance
			if(null == dist_mat)
				throw new IllegalStateException("null distance matrix");
		}
		
		abstract double[][] link();
		abstract double[][] mutualReachability();
	}
	
	abstract class KDTree extends HDBSCANLinkageTree {
		int leafSize;
		
		KDTree(int leafSize) {
			super();
			
			this.leafSize = leafSize;
			Class<? extends GeometricallySeparable> clz = 
				model.getSeparabilityMetric().getClass();
			if( !ValidKDMetrics.contains(clz) ) {
				warn(clz+" is not a valid distance metric for KDTrees. Valid metrics: " + ValidKDMetrics);
				warn("falling back to default metric: " + AbstractClusterer.DEF_DIST);
				model.setSeparabilityMetric(AbstractClusterer.DEF_DIST);
			}
		}
	}
	
	abstract class BallTree extends HDBSCANLinkageTree {
		BallTree() {
			super();
		}
		
		final double[][] mutualReachability() {
			// TODO:
			return null;
		}
	}
	
	/**
	 * Generic single linkage tree
	 * @author Taylor G Smith
	 */
	class GenericTree extends HDBSCANLinkageTree {
		GenericTree() {
			super();
		}
		
		@Override
		double[][] link() {
			final double[][] mutual_reachability = mutualReachability();
			double[][] min_spanning_tree = LinkageTreeUtils.mstLinkageCore(mutual_reachability, m);
			
			// Sort edges of the min_spanning_tree by weight
			min_spanning_tree = MatUtils.sortAscByCol(min_spanning_tree, 2);
			return label(min_spanning_tree);
		}
		
		@Override
		double[][] mutualReachability() { // Tested: passing
			final int min_points = FastMath.min(m - 1, minPts);
			final double[] core_distances = MatUtils
				.partitionByRow(dist_mat, min_points)[min_points];
			
			if(alpha != 1.0)
				dist_mat = MatUtils.scalarDivide(dist_mat, alpha);
			
			
			final MatSeries ser1 = new MatSeries(core_distances, Inequality.GT, dist_mat);
			double[][] stage1 = MatUtils.where(ser1, core_distances, dist_mat);
			
			stage1 = MatUtils.transpose(stage1);
			final MatSeries ser2 = new MatSeries(core_distances, Inequality.GT, stage1);
			final double[][] result = MatUtils.where(ser2, core_distances, stage1);
			
			return MatUtils.transpose(result);
		}
	}
	
	class PrimsKDTree extends KDTree implements Prim {
		PrimsKDTree(int leafSize) {
			super(leafSize);
		}
		
		@Override
		double[][] link() {
			final int min_points = FastMath.min(m - 1, minPts);
			BinaryKDTree tree = new BinaryKDTree(data, leafSize, model);
			// TODO
			
			return null;
		}
		
		@Override
		double[][] mutualReachability() {
			// TODO:
			return null;
		}

		@Override
		public double[][] mstLinkageCore(double[][] X, double[][] coreDists) {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	class PrimsBallTree extends BallTree implements Prim {
		PrimsBallTree() {
			super();
		}

		@Override
		public double[][] mstLinkageCore(double[][] X, double[][] coreDists) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		double[][] link() {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	class BoruvkaKDTree extends KDTree implements Boruvka {
		BoruvkaKDTree(int leafSize) {
			super(leafSize);
		}

		@Override
		double[][] link() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		double[][] mutualReachability() {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	class BoruvkaBallTree extends BallTree implements Boruvka {
		BoruvkaBallTree() {
			super();
		}

		@Override
		double[][] link() {
			// TODO Auto-generated method stub
			return null;
		}
	}
	
	interface UnifiedFinder {
		void union(int m, int n);
		int find(int x);
	}
	
	static class TreeUnifyFind implements UnifiedFinder {
		int size;
		int [][] dataArr;
		int [] is_component;
		
		public TreeUnifyFind(int size) {
			dataArr = new int[size][2];
			// First col should be arange to size
			for(int i = 0; i < size; i++)
				dataArr[i][0] = i;
			
			is_component = VecUtils.repInt(1, size);
			this.size= size;
		}
		
		@Override
		public void union(int x, int y) {
			int x_root = find(x);
			int y_root = find(y);
			
			int x1idx = LinkageTreeUtils.wraparoundIdxGet(size, x_root);
			int y1idx = LinkageTreeUtils.wraparoundIdxGet(size, x_root);
			
			int dx1 = dataArr[x1idx][1];
			int dy1 = dataArr[y1idx][1];
			
			if(dx1 < dy1)
				dataArr[x1idx][0] = y_root;
			else if(dx1 > dy1)
				dataArr[y1idx][0] = x_root;
			else {
				dataArr[y1idx][0] = x_root;
				dataArr[x1idx][1] += 1;
			}
		}
		
		@Override
		public int find(int x) {
			final int idx = LinkageTreeUtils.wraparoundIdxGet(size, x);
			if(dataArr[idx][0] != x) {
				dataArr[idx][0] = find(dataArr[idx][0]);
				is_component[idx] = 0;
			}
			
			return dataArr[idx][0];
		}
		
		/**
		 * Returns all non-zero indices in is_component
		 * @return
		 */
		int[] components() {
			final HList<Integer> h = new HList<>();
			for(int i: is_component)
				if(i == 1)
					h.add(i);
			
			int idx = 0;
			int[] out = new int[h.size()];
			for(Integer i: h)
				out[idx++] = i;
			
			return out;
		}
	}
	
	static class UnifyFind implements UnifiedFinder {
		int [] parentArr, sizeArr, parent, size;
		int nextLabel;
		
		public UnifyFind(int N) {
			parentArr = VecUtils.repInt(-1, 2 * N - 1);
			nextLabel = N;
			sizeArr = VecUtils.cat(
					VecUtils.repInt(1, N), 
					VecUtils.repInt(0, N-1));
			
			parent = parentArr;
			size = sizeArr;
		}
		
		int fastFind(int n) {
			int p = n, tmp;
			while(parentArr[n] != -1)
				n = parentArr[n];
			
			while(parentArr[LinkageTreeUtils.wraparoundIdxGet(parentArr.length, p)] != n) {
				tmp = parentArr[LinkageTreeUtils.wraparoundIdxGet(parentArr.length,p)];
				
				parentArr[LinkageTreeUtils.wraparoundIdxGet(parentArr.length, p)] = n;
				p = tmp;
			}
			
			return n;
		}
		
		@Override
		public int find(int n) {
			while(parent[n] != -1)
				n = parent[n];
			return n;
		}
		
		@Override
		public void union(final int m, final int n) {
			size[nextLabel] = size[m] + size[n];
			parent[m] = nextLabel;
			parent[n] = nextLabel;
			size[nextLabel] = size[m] + size[n];
			nextLabel++;
		}
		
		@Override
		public String toString() {
			return "Parent arr: " + Arrays.toString(parentArr) + "; " +
					"Sizes: " + Arrays.toString(sizeArr) + "; " +
					"Parent: " + Arrays.toString(parent);
		}
	}
	
	

	


	protected static int[] doLabeling(HList<QuadTup<Integer, Integer, Double, Integer>> tree,
			HList<Integer> clusters, TreeMap<Integer, Integer> clusterMap) {
		
		QuadTup<Integer, Integer, Double, Integer> quad;
		int rootCluster, parent, child, n = tree.size(), cluster, i;
		int[] resultArr, parentArr = new int[n], childArr = new int[n];
		UnifiedFinder unionFind;
		
		// [parent, child, lambda, size]
		int maxParent = Integer.MIN_VALUE;
		int minParent = Integer.MAX_VALUE;
		for(i = 0; i < n; i++) {
			quad = tree.get(i);
			parentArr[i]= quad.one;
			childArr[i] = quad.two;
			
			if(quad.one < minParent)
				minParent = quad.one;
			if(quad.one > maxParent)
				maxParent = quad.one;
		}
		
		rootCluster = minParent;
		resultArr = new int[rootCluster];
		unionFind = new TreeUnifyFind(maxParent + 1);
		
		for(i = 0; i < n; i++) {
			child = childArr[i];
			parent= parentArr[i];
			if(!clusters.contains(child))
				unionFind.union(parent, child);
		}
		
		for(i = 0; i < rootCluster; i++) {
			cluster = unionFind.find(i);
			if(cluster <= rootCluster)
				resultArr[i] = NOISE_CLASS;
			else
				resultArr[i] = clusterMap.get(cluster);
		}
		
		return resultArr;
	}
	
	@Override
	public HDBSCAN fit() {
		synchronized(this) {
			
			try {
				if(null!=labels) // Then we've already fit this...
					return this;
				
				// First get the dist matrix
				final long start = System.currentTimeMillis();
				info("fitting model");
				dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
				
				
				// Build the tree
				String msg = "constructing HDBSCAN single linkage dendrogram: ";
				Class<? extends HDBSCANLinkageTree> clz = null;
				switch(algo) {
					case GENERIC:
						clz = GenericTree.class;
						info(msg + clz.getName());
						tree = new GenericTree();
						break;
					/*case PRIMS_KD_TREE:
						clz = PrimsKDTree.class;
						info(msg + clz.getName());
						tree = new PrimsKDTree(leafSize);
						break;*/
					default:
						throw new InternalError("illegal algorithm");
				}
				
				final double[][] build = tree.link();
				
				info("Labeling clusters");
				labels = treeToLabels(data.getData(), build, min_cluster_size);
				
				info("model "+getKey()+" completed in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
						System.lineSeparator());
				
				dist_mat = null;
				tree = null;
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
		}
	}

	
	@Override
	public int[] getLabels() {
		try {
			return VecUtils.copy(labels);
		} catch(NullPointerException npe) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
		}
	}

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.HDBSCAN;
	}

	@Override
	public String getName() {
		return "HDBSCAN";
	}

	@Override
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}

	@Override
	public int getNumberOfNoisePoints() {
		return numNoisey;
	}
	
	protected static int[] getLabels(HList<QuadTup<Integer, Integer, Double, Integer>> condensed,
									TreeMap<Integer, Double> stability) {
		
		double subTreeStability;
		double[][] tmpClusterTree;
		int parent;
		HList<Integer> nodes, clusters;
		
		// Get descending sorted key set
		int ct = 0;
		HList<Integer> nodeList = new HList<>();
		for(Integer d: stability.descendingKeySet())
			if(++ct < stability.size()) // exclude the root...
				nodeList.add(d);
		
		
		// Within this list, save which nodes map to parents that have the same value as the node...
		TreeMap<Integer, HList<QuadTup<Integer, Integer, Double, Integer>>> nodeMap = new TreeMap<>();
		
		// [parent, child, lambda, size]
		int maxChildSize = Integer.MIN_VALUE;
		HList<QuadTup<Integer, Integer, Double, Integer>> clusterTree = new HList<>();
		for(QuadTup<Integer, Integer, Double, Integer> branch: condensed) {
			parent = branch.one;
			if(!nodeMap.containsKey(parent))
				nodeMap.put(parent, new HList<QuadTup<Integer, Integer, Double, Integer>>());
			nodeMap.get(parent).add(branch);
			
			if(branch.four > 1) // where childSize > 1
				clusterTree.add(branch);
			else if(branch.four == 1) {
				if(branch.two > maxChildSize)
					maxChildSize = branch.two;
			}
		}
		
		// Build the tmp cluster tree
		tmpClusterTree = new double[clusterTree.size()][4];
		for(int i = 0; i < tmpClusterTree.length; i++) {
			tmpClusterTree[i] = new double[]{
				clusterTree.get(i).one,
				clusterTree.get(i).two,
				clusterTree.get(i).three,
				clusterTree.get(i).four,
			};
		}
		
		// Get cluster TreeMap
		TreeMap<Integer, Boolean> isCluster = new TreeMap<>();
		for(Integer d: nodeList) // init as true
			isCluster.put(d, true);
		
		// Big loop
		HList<QuadTup<Integer, Integer, Double, Integer>> childSelection;
		//int numPoints = maxChildSize + 1;
		for(Integer node: nodeList) {
			childSelection = nodeMap.get(node);
			subTreeStability = 0;
			if(null != childSelection)
				for(QuadTup<Integer,Integer,Double,Integer> selection: childSelection) {
					subTreeStability += stability.get(selection.two);
				}
			
			if(subTreeStability > stability.get(new Double(node))) {
				isCluster.put(node, false);
				stability.put(node, subTreeStability);
			} else {
				nodes = LinkageTreeUtils.breadthFirstSearch(tmpClusterTree, node);
				for(Integer subNode: nodes)
					if(subNode != node)
						isCluster.put(subNode, false);
			}
		}
		
		// Set clusters
		clusters = new HList<>();
		for(Map.Entry<Integer, Boolean> entry: isCluster.entrySet())
			if(entry.getValue())
				clusters.add(entry.getKey());
		
		// Enumerate clusters
		TreeMap<Integer, Integer> reverseClusterMap = new TreeMap<>();
		TreeMap<Integer, Integer> clusterMap = new TreeMap<>();
		for(int n = 0; n < clusters.size(); n++) {
			clusterMap.put(n, clusters.get(n));
			reverseClusterMap.put(clusters.get(n), n);
		}
		
		return doLabeling(condensed, clusters, clusterMap);
	}
	
	protected static double[][] label(final double[][] tree) {
		double[][] result;
		int a, aa, b, bb, index;
		final int m = tree.length, n = tree[0].length, N = m + 1;
		double delta;
		
		result = new double[m][n+1];
		UnifyFind U = new UnifyFind(N);
		
		
		for(index = 0; index < m; index++) {
			
			a = (int)tree[index][0];
			b = (int)tree[index][1];
			delta = tree[index][2];
			
			aa = U.fastFind(a);
			bb = U.fastFind(b);
			
			result[index][0] = aa;
			result[index][1] = bb;
			result[index][2] = delta;
			result[index][3] = U.size[aa] + U.size[bb];
			
			U.union(aa, bb);
		}
		
		return result;
	}
	
	protected static double[][] singleLinkage(final double[][] dists) {
		final double[][] hierarchy = LinkageTreeUtils.mstLinkageCore(dists, dists.length);
		return label(MatUtils.sortAscByCol(hierarchy, 2));
	}
	
	protected static int[] treeToLabels(final double[][] X, 
			final double[][] single_linkage_tree, final int min_size) {
		
		final HList<QuadTup<Integer, Integer, Double, Integer>> condensed = 
				LinkageTreeUtils.condenseTree(single_linkage_tree, min_size);
		final TreeMap<Integer, Double> stability = LinkageTreeUtils.computeStability(condensed);
		
		return getLabels(condensed, stability);
	}
}
