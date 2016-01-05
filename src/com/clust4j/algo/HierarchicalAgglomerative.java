package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.Loggable;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.IllegalClusterStateException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

/**
 * Agglomerative clustering is a hierarchical clustering process in
 * which each input record initially is mapped to its own cluster.
 * Progressively, each cluster is merged by locating the least dissimilar 
 * clusters in a M x M distance matrix, merging them, removing the corresponding
 * rows and columns from the distance matrix and adding a new row/column vector
 * of distances corresponding to the new cluster until there is one cluster.
 * <p>
 * Agglomerative clustering does <i>not</i> scale well to large data, performing
 * at O(n<sup>2</sup>) computationally, yet it outperforms its cousin, Divisive Clustering 
 * (DIANA), which performs at O(2<sup>n</sup>).
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 * 
 * @see com.clust4j.utils.SingleLinkageAgglomerativeFactory
 * @see com.clust4j.utils.HierarchicalClusterTree
 * @see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html">Agglomerative Clustering</a>
 * @see <a href="http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm">Divisive Clustering</a>
 */
public class HierarchicalAgglomerative extends AbstractPartitionalClusterer implements Classifier {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7563413590708853735L;
	public static final Linkage DEF_LINKAGE = Linkage.WARD;
	
	/**
	 * Which {@link Linkage} to use for the clustering algorithm
	 */
	final private Linkage linkage;
	/**
	 * The number of rows in the matrix
	 */
	final private int m;
	
	
	/**
	 * The linkages for agglomerative clustering. 
	 * @author Taylor G Smith
	 *
	 */
	public enum Linkage implements java.io.Serializable {
		AVERAGE, COMPLETE, WARD
	}
	
	
	/**
	 * The labels for the clusters
	 */
	volatile private int[] labels = null;
	/**
	 * The distance matrix. Set to null after fit
	 */
	volatile private double[][] dist_mat = null;
	volatile HierarchicalDendrogram tree = null;
	/** 
	 * Volatile because if null will later change during build
	 */
	volatile private int num_clusters;
	
	
	
	
	
	
	public HierarchicalAgglomerative(AbstractRealMatrix data) {
		this(data, new HierarchicalPlanner());
	}

	public HierarchicalAgglomerative(AbstractRealMatrix data, 
			HierarchicalPlanner planner) {
		super(data, planner, planner.num_clusters);
		
		this.linkage = planner.linkage;
		if(null == linkage) {
			String e = "null linkage passed to planner";
			error(e);
			throw new IllegalClusterStateException(e);
		} else if(linkage.equals(Linkage.WARD) && !getSeparabilityMetric().equals(Distance.EUCLIDEAN)) {
			warn("Ward's method implicitly requires Euclidean distance; overriding " + 
				getSeparabilityMetric().getName());
			
			super.setSeparabilityMetric(Distance.EUCLIDEAN);
			meta("New distance metric: "+getSeparabilityMetric().getName());
		}
		
		
		
		this.m = data.getRowDimension();
		this.num_clusters = planner.num_clusters;
		
		
		meta("Linkage="+linkage);
		meta("Num clusters="+num_clusters);
		trace(getName()+" clustering has a runtime of O(N^2)");
	}
	
	
	
	
	
	public static class HierarchicalPlanner extends AbstractClusterer.BaseClustererPlanner {
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private Linkage linkage = DEF_LINKAGE;
		private boolean verbose = DEF_VERBOSE;
		private int num_clusters = 2;
		private FeatureNormalization norm = DEF_NORMALIZER;
		
		
		public HierarchicalPlanner(){}
		public HierarchicalPlanner(Linkage linkage) {
			this();
			this.linkage = linkage;
		}
		

		@Override
		public HierarchicalAgglomerative buildNewModelInstance(AbstractRealMatrix data) {
			return new HierarchicalAgglomerative(data, this);
		}
		
		@Override
		public HierarchicalPlanner copy() {
			return new HierarchicalPlanner(linkage)
				.setSep(dist)
				.setScale(scale)
				.setSeed(seed)
				.setVerbose(verbose)
				.setNumClusters(num_clusters)
				.setNormalizer(norm);
		}
		
		public Linkage getLinkage() {
			return linkage;
		}

		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}

		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		public HierarchicalPlanner setLinkage(Linkage l) {
			this.linkage = l;
			return this;
		}
		
		public HierarchicalPlanner setNumClusters(final int d) {
			this.num_clusters = d;
			return this;
		}

		@Override
		public HierarchicalPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}
		
		@Override
		public HierarchicalPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public HierarchicalPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}

		@Override
		public HierarchicalPlanner setSep(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}
		@Override
		public HierarchicalPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	
	
	
	
	abstract class HierarchicalDendrogram implements java.io.Serializable {
		private static final long serialVersionUID = 5295537901834851676L;
		public final Loggable ref;
		public final GeometricallySeparable dist;
		public final int m;
		private final double[] dist_vec;
		
		HierarchicalDendrogram() {
			if(null == dist_mat)
				dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
			
			ref = HierarchicalAgglomerative.this;
			dist = HierarchicalAgglomerative.this.getSeparabilityMetric();
			m = HierarchicalAgglomerative.this.m;
			
			// Flatten upper triangular dist_mat
			final int s = m*(m-1)/2; // The shape of the flattened upper triangular matrix (m choose 2)
			dist_vec = new double[s];
			for(int i = 0, r = 0; i < m - 1; i++)
				for(int j = i + 1; j < m; j++, r++)
					dist_vec[r] = dist_mat[i][j];
		}
		
		double[][] linkage() {
			// Perform the linkage logic in the tree
			double[] y = VecUtils.copy(dist_vec); // Copy the dist_vec
			
			double[][] Z = new double[m - 1][4];  // Holding matrix
			link(y, Z, m); // Immutabily change Z
			
			// Final linkage tree out...
			return MatUtils.getColumns(Z, new int[]{0,1});
		}
		
		private void link(final double[] dists, final double[][] Z, final int n) {
			int i, j, k, x = -1, y = -1, i_start, nx, ny, ni, id_x, id_y, id_i, c_idx;
			double current_min;
			
			// Inter cluster dists
			double[] D = VecUtils.copy(dists);
			
			// Map the indices to node ids
			ref.info("initializing node mappings ("+getClass().getName().split("\\$")[1]+")");
			int[] id_map = new int[n];
			for(i = 0; i < n; i++) 
				id_map[i] = i;
			
			long start = System.currentTimeMillis();
			int incrementor = n/5, pct = 1;
			for(k = 0; k < n - 1; k++) {
				if(incrementor>0 && k%incrementor == 0)
					ref.info("node mapping progress - " + 20*pct++ + "% ("+
						LogTimeFormatter.millis( System.currentTimeMillis()-start , false)+")");
				
				// get two closest x, y
				current_min = Double.POSITIVE_INFINITY;
				
				for(i = 0; i < n - 1; i++) {
					if(id_map[i] == -1)
						continue;
					
					
					i_start = condensedIndex(n, i, i + 1);
					for(j = 0; j < n - i - 1; j++) {
						if(D[i_start + j] < current_min) {
							current_min = D[i_start + j];
							x = i;
							y = i + j + 1;
						}
					}
				}
				
				id_x = id_map[x];
				id_y = id_map[y];
				
				// Get original num points in clusters x,y
				nx = id_x < n ? 1 : (int)Z[id_x - n][3];
				ny = id_y < n ? 1 : (int)Z[id_y - n][3];
				
				// Record new node
				Z[k][0] = FastMath.min(id_x, id_y);
				Z[k][1] = FastMath.max(id_y, id_x);
				Z[k][2] = current_min;
				Z[k][3] = nx + ny;
				id_map[x] = -1; // cluster x to be dropped
				id_map[y] = n + k; // cluster y replaced
				
				// update dist mat
				for(i = 0; i < n; i++) {
					id_i = id_map[i];
					if(id_i == -1 || id_i == n + k)
						continue;
					
					ni = id_i < n ? 1 : (int)Z[id_i - n][3];
					c_idx = condensedIndex(n, i, y);
					D[c_idx] = getDist(D[condensedIndex(n, i, x)],
						D[c_idx], current_min, nx, ny, ni);
					
					if(i < x)
						D[condensedIndex(n,i,x)] = Double.POSITIVE_INFINITY;
				}
			}
		}
		
		abstract protected double getDist(final double dx, final double dy, 
			final double current_min, final int nx, final int ny, final int ni);
	}
	
	class WardTree extends HierarchicalDendrogram {
		private static final long serialVersionUID = -2336170779406847047L;
		
		public WardTree() { super(); }

		@Override
		protected double getDist(double dx, double dy, 
			double current_min, int nx, int ny, int ni) {
			
			final double t = 1.0 / (nx + ny + ni);
			return FastMath.sqrt((ni + nx) * t * dx * dx +
								 (ni + ny) * t * dy * dy -
								 ni * t * current_min * current_min);
		}
	}
	
	abstract class LinkageTree extends HierarchicalDendrogram {
		private static final long serialVersionUID = -252115690411913842L;
		public LinkageTree() { super(); }
	}
	
	class AverageLinkageTree extends LinkageTree {
		private static final long serialVersionUID = 5891407873391751152L;

		public AverageLinkageTree() { super(); }
		
		@Override
		protected double getDist(double dx, double dy, 
			double current_min, int nx, int ny, int ni) {
			return (nx * dx + ny * dy) / (double)(nx + ny);
		}
	}
	
	class CompleteLinkageTree extends LinkageTree {
		private static final long serialVersionUID = 7407993870975009576L;
		
		public CompleteLinkageTree() { super(); }

		@Override
		protected double getDist(double dx, double dy, 
			double current_min, int nx, int ny, int ni) {
			return FastMath.max(dx, dy);
		}
	}
	
	/**
	 * Heapifies an ArrayList in place. Adapted from Python's
	 * <a href="https://github.com/python-git/python/blob/master/Lib/heapq.py">heapq</a>
	 * priority queue.
	 * @author Taylor G Smith
	 */
	static class HeapUtils {
		public static <T extends Comparable<? super T>> void heapifyInPlace(final ArrayList<T> x) {
			final int n = x.size();
			final int n_2_floor = n / 2;
			
			for(int i = n_2_floor - 1; i >= 0; i--)
				siftUp(x, i);
		}
		
		public static <T extends Comparable<? super T>> T heapPop(ArrayList<T> heap) {
			final T lastElement = popInPlace(heap), returnItem;
			
			if(heap.size() > 0) {
				returnItem = heap.get(0);
				heap.set(0, lastElement);
				siftUp(heap, 0);
			} else {
				returnItem = lastElement;
			}
			
			return returnItem;
		}
		
		public static <T extends Comparable<? super T>> void heapPush(final ArrayList<T> heap, T item) {
			heap.add(item);
			siftDown(heap, 0, heap.size()-1);
		}
		
		public static <T extends Comparable<? super T>> T heapPushPop(final ArrayList<T> heap, T item) {
			if(heap.get(0).compareTo(item) < 0) {
				T tmp = heap.get(0);
				heap.set(0, item);
				item = tmp;
			}
			
			return item;
		}
		
		public static <T extends Comparable<? super T>> T heapReplace(final ArrayList<T> heap, T item) {
			T returnItem = heap.get(0);
			heap.set(0, item);
			siftUp(heap, 0);
			return returnItem;
		}
		
		public static <T extends Comparable<? super T>> ArrayList<T> nLargest(final int n, final Collection<T> iterable) {
			if(n >= iterable.size())
				throw new IllegalArgumentException();
			
			final ArrayList<T> copy = new ArrayList<T>();
			for(T t: iterable)
				copy.add(t);
			
			Collections.sort(copy);
			Collections.reverse(copy);
			
			return (ArrayList<T>) copy.subList(0, n);
		}
		
		public static <T extends Comparable<? super T>> ArrayList<T> nSmallest(final int n, final Collection<T> iterable) {
			if(n >= iterable.size())
				throw new IllegalArgumentException();
			
			final ArrayList<T> copy = new ArrayList<T>();
			for(T t: iterable)
				copy.add(t);
			
			Collections.sort(copy);
			
			return (ArrayList<T>) copy.subList(0, n);
		}
		
		static <T extends Comparable<? super T>> T popInPlace(final ArrayList<T> heap) {
			if(heap.size() == 0)
				throw new InternalError("heap size 0");
			
			final T last = heap.get(heap.size()-1);
			heap.remove(heap.size()-1);
			return last;
		}
		
		static <T extends Comparable<? super T>> void siftDown(final ArrayList<T> heap, final int startPos, int pos) {
			T newitem = heap.get(pos);
			
			while(pos > startPos) {
				int parentPos = (pos - 1) >> 1;
				T parent = heap.get(parentPos);
				
				if(newitem.compareTo(parent) < 0) {
					heap.set(pos, parent);
					pos = parentPos;
					continue;
				}
				
				break;
			}
			
			heap.set(pos, newitem);
		}
		
		static <T extends Comparable<? super T>> void siftUp(final ArrayList<T> heap, int pos) {
			int endPos = heap.size();
			int startPos= pos;
			T newItem = heap.get(pos);
			
			int childPos = 2*pos + 1;
			while(childPos < endPos) {
				int rightPos = childPos + 1;
				if(rightPos < endPos && !(heap.get(childPos).compareTo(heap.get(rightPos)) < 0))
					childPos = rightPos;
				
				heap.set(pos, heap.get(childPos));
				pos = childPos;
				childPos = 2*pos + 1;
			}
			
			heap.set(pos, newItem);
			siftDown(heap, startPos, pos);
		}
	}
	
	
	

	
	/**
	 * Get condensed index from 
	 * collapsed upper triangular matrix
	 * @param n
	 * @param i
	 * @param j
	 * @return
	 */
	static int condensedIndex(int n, int i, int j) {
		if(i < j)
			return n * i - (i * (i + 1) / 2) + (j - i - 1);
		else if(i > j)
			return n * j - (j * (j + 1) / 2) + (i - j - 1);
		throw new InternalError(i+", "+j);
	}
	
	
	@Override
	public String getName() {
		return "Agglomerative";
	}

	@Override
	public HierarchicalAgglomerative fit() {
		synchronized(this) { // synch because alters internal structs
			
			if(null != labels) // Then we've already fit this...
				return this;
			
			try {
				final long start = System.currentTimeMillis();
				labels = new int[m];
				dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
				
				// Log info...
				info("calculated " + 
					m + " x " + m + 
					" distance matrix in " + 
					LogTimeFormatter.millis( System.currentTimeMillis()-start , false));
				
				
				
				// Get the tree class for logging...
				Class<? extends HierarchicalDendrogram> clz = null;
				switch(linkage) {
					case WARD:
						clz = WardTree.class;
						info("constructing HierarchicalDendrogram: " + clz.getName());
						tree = new WardTree();
						break;
					case AVERAGE:
						clz = AverageLinkageTree.class;
						info("constructing HierarchicalDendrogram: " + clz.getName());
						tree = new AverageLinkageTree();
						break;
					case COMPLETE:
						clz = CompleteLinkageTree.class;
						info("constructing HierarchicalDendrogram: " + clz.getName());
						tree = new CompleteLinkageTree();
						break;
					default:
						throw new InternalError("illegal linkage");
				}
				
				
				// Tree build
				double[][] children = tree.linkage();
				
				
				
				// Cut the tree
				labels = hcCut(num_clusters, children, m);
				reorderLabels();
				
				
				info("model " + getKey() + " completed in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-start, false) +
						System.lineSeparator());
				
				dist_mat = null;
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			}
			
		} // End synch
	} // End train
	
	static int[] hcCut(final int n_clusters, final double[][] children, final int n_leaves) {
		/*
		 * Leave children as a double[][] despite it
		 * being ints. This will allow VecUtils to operate
		 */
		
		if(n_clusters > n_leaves)
			throw new InternalError(n_clusters + " > " + n_leaves);
		
		// Init nodes
		ArrayList<Integer> nodes = new ArrayList<>(Arrays.asList(new Integer[]{-((int)VecUtils.max(children[children.length-1]) + 1)}));

		
		for(int i = 0; i < n_clusters - 1; i++) {
			int inner_idx = -nodes.get(0) - n_leaves;
			if(inner_idx < 0)
				inner_idx = children.length + inner_idx;
			
			double[] these_children = children[inner_idx];
			HeapUtils.heapPush(nodes, -((int)these_children[0]));
			HeapUtils.heapPushPop(nodes, -((int)these_children[1]));
		}

		int i = 0;
		final int[] labels = new int[n_leaves];
		for(Integer node: nodes) {
			Integer[] descendants = hcGetDescendents(-node, children, n_leaves);
			for(Integer desc: descendants)
				labels[desc] = i;
			
			i++;
		}
		
		return labels;
	}
	
	static Integer[] hcGetDescendents(int node, double[][] children, int leaves) {
		final ArrayList<Integer> ind = new ArrayList<>(Arrays.asList(new Integer[]{node}));
		if(node < leaves)
			return new Integer[]{node};
		
		final ArrayList<Integer> descendent = new ArrayList<>();
		int i, n_indices = 1;
		
		while(n_indices > 0) {
			i = HeapUtils.popInPlace(ind);
			if(i < leaves) {
				descendent.add(i);
				n_indices--;
			} else {
				final double[] chils = children[i - leaves];
				for(double d: chils)
					ind.add((int)d);
				n_indices++;
			}
		}
		
		return descendent.toArray(new Integer[descendent.size()]);
	}
	
	public int[] getLabels() {
		try {
			return VecUtils.copy(labels);
		} catch(NullPointerException e) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
		}
	}
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AGGLOMERATIVE;
	}
	
	final private void reorderLabels() {
		// Now rearrange labels in order... first get unique labels in order of appearance
		final ArrayList<Integer> orderOfLabels = new ArrayList<Integer>(k);
		for(int label: labels) {
			if(!orderOfLabels.contains(label)) // Race condition? but synchronized so should be ok...
				orderOfLabels.add(label);
		}
		
		final int[] newLabels = new int[m];
		for(int i = 0; i < m; i++)
			newLabels[i] = orderOfLabels.indexOf(labels[i]);
		
		// Reassign labels...
		labels = newLabels;
	}
}
