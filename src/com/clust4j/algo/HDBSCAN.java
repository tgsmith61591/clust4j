package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.utils.QuadTup;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.DistanceMetric;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.IllegalClusterStateException;
import com.clust4j.utils.Inequality;
import com.clust4j.utils.MatUtils;
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
	//private final boolean genMinSpanTree; // NEVER

	private volatile HDBSCANLinkageTree tree = null;
	private volatile double[][] dist_mat = null;
	private volatile int[] labels = null;
	private volatile int numClusters;
	private volatile int numNoisey;
	/** A copy of the data array inside the data matrix */
	private volatile double[][] dataData = null;
	
	
	public static enum Algorithm {
		GENERIC,
		PRIMS_KDTREE,
		PRIMS_BALLTREE,
		//BORUVKA_KDTREE,
		//BORUVKA_BALLTREE
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
		
		
		if(!algo.equals(Algorithm.GENERIC) && !(planner.getSep() instanceof DistanceMetric)) {
			warn("algorithms leveraging NearestNeighborHeapSearch require a DistanceMetric; "
					+ "using default Euclidean distance");
			setSeparabilityMetric(DEF_DIST);
		}
		
		logModelSummary();
	}
	
	@Override
	String modelSummary() {
		final ArrayList<Object[]> formattable = new ArrayList<>();
		formattable.add(new Object[]{
			"Num Rows","Num Cols","Metric","Algo.","Scale","Force Par.","Allow Par.","Min Pts.","Min Clust. Size","Alpha"
		});
		
		formattable.add(new Object[]{
			data.getRowDimension(),data.getColumnDimension(),
			getSeparabilityMetric(),algo,normalized,
			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE,
			GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM,
			minPts, min_cluster_size,alpha
		});
		
		return formatter.format(formattable);
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
		
		
		public HDBSCANPlanner() { this(DEF_MIN_PTS); }
		public HDBSCANPlanner(final int minPts) {
			this.minPts = minPts;
		}

		
		@Override
		public HDBSCAN buildNewModelInstance(AbstractRealMatrix data) {
			return new HDBSCAN(data, this.copy());
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
	 * A simple extension of {@link ArrayList} that also provides
	 * a simple heap/stack interface of methods.
	 * @author Taylor G Smith
	 * @param <T>
	 */
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
	
	/**
	 * A simple extension of {@link HashSet} that takes
	 * an array or varargs as a constructor arg
	 * @author Taylor G Smith
	 * @param <T>
	 */
	final static class HSet<T> extends HashSet<T> {
		private static final long serialVersionUID = 5185550036712184095L;
		
		HSet(int size) {
			super(size);
		}
		
		HSet(T[] args) {
			this(args.length);
			for(T t: args)
				add(t);
		}
		
		HSet(Collection<? extends T> coll) {
			super(coll);
		}
	}
	
	/**
	 * Constructs an {@link HSet} from the labels
	 * @author Taylor G Smith
	 */
	final static class LabelHSetFactory {
		static HSet<Integer> build(int[] labs) {
			HSet<Integer> res = new HSet<Integer>(labs.length);
			for(int i: labs)
				res.add(i);
			
			return res;
		}
	}
	
	
	/** Classes that will explicitly need to define 
	 *  reachability will have to implement this interface */
	interface ExplicitMutualReachability { double[][] mutualReachability(); }
	/**
	 * Mutual reachability is implicit when using 
	 * {@link BoruvkaAlgorithm},
	 * thus we don't need these classes to implement 
	 * {@link ExplicitMutualReachability#mutualReachability()} */
	interface Boruvka {}
	/**
	 * Mutual reachability is implicit when using 
	 * {@link LinkageTreeUtils#mstLinkageCore_cdist},
	 * thus we don't need these classes to implement 
	 * {@link ExplicitMutualReachability#mutualReachability()} */
	interface Prim {}
	
	
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
			int root = 2 * m, 
					numPoints = root/2 + 1 /*Integer division*/, 
					nextLabel = numPoints+1;
			
			// Get node list from BFS
			HList<Integer> nodeList = breadthFirstSearch(hierarchy, root), tmpList;
			HList<QuadTup<Integer, Integer, Double, Integer>> resultList = new HList<>();
			
			// Indices needing relabeling -- cython code assigns this to nodeList.size()
			// but often times this is way too small and causes out of bounds exceptions...
			// Changed to root + 1 on 02/01/2016; this should be the max node ever in the resultList
			int[] relabel = new int[root + 1]; //nodeList.size()
			boolean[] ignore = new boolean[root + 1];
			double[] children;
			
			double lambda;
			int left, right, leftCount, rightCount;
			
			// Sanity check
			// System.out.println("Root: " + root + ", Relabel length: " + relabel.length + ", m: " + m + ", Relabel array: " + Arrays.toString(relabel));
			
			// The cython code doesn't check for bounds and sloppily 
			// assigns this even if root > relabel.length. 
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
		static double[][] minSpanTreeLinkageCore(final double[][] X, final int m) { // Tested: passing
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
		
		static double[][] minSpanTreeLinkageCore_cdist(final double[][] raw, final double[] coreDistances, GeometricallySeparable sep, final double alpha) {
			double[] currentDists;
			int[] inTreeArr;
			double[][] resultArr;
			
			int currentNode = 0, newNode, i, j, dim = raw.length;
			double currentNodeCoreDist, rightVal, leftVal, coreVal, newDist;
			
			resultArr = new double[dim - 1][3];
			inTreeArr = new int[dim];
			currentDists = VecUtils.rep(Double.POSITIVE_INFINITY, dim);
			
			
			for(i = 1; i < dim; i++) {
				inTreeArr[currentNode] = 1;
				currentNodeCoreDist = coreDistances[currentNode];
				
				newDist = Double.MAX_VALUE;
				newNode = 0;
				
				for(j = 0; j < dim; j++) {
					if(inTreeArr[j] != 0)
						continue; // only skips currentNode idx
					
					rightVal = currentDists[j];
					leftVal = sep.getDistance(raw[currentNode], raw[j]);
					
					if(alpha != 1.0)
						leftVal /= alpha;
					
					coreVal = coreDistances[j];
					if(currentNodeCoreDist > rightVal || coreVal > rightVal
						|| leftVal > rightVal) {
						if(rightVal < newDist) { // Should always be the case?
							newDist = rightVal;
							newNode = j;
						}
						
						continue;
					}
					
					
					if(coreVal > currentNodeCoreDist) {
						if(coreVal > leftVal)
							leftVal = coreVal;
					} else {
						if(currentNodeCoreDist > leftVal)
							leftVal = currentNodeCoreDist;
					}
					
					
					if(leftVal < rightVal) {
						currentDists[j] = leftVal;
						if(leftVal < newDist) {
							newDist = leftVal;
							newNode = j;
						}
					} else {
						if(rightVal < newDist) {
							newDist = rightVal;
							newNode = j;
						}
					}
				} // end for j
				
				resultArr[i - 1][0] = currentNode;
				resultArr[i - 1][1] = newNode;
				resultArr[i - 1][2] = newDist;
				currentNode = newNode;
			} // end for i
			
			
			return resultArr;
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
	
	
	/**
	 * The top level class for all HDBSCAN linkage trees.
	 * @author Taylor G Smith
	 */
	abstract class HDBSCANLinkageTree {
		final HDBSCAN model;
		final GeometricallySeparable metric;
		final int m, n;
		
		HDBSCANLinkageTree() {
			model = HDBSCAN.this;
			metric = model.getSeparabilityMetric();
			m = model.data.getRowDimension();
			n = model.data.getColumnDimension();
		}
		
		abstract double[][] link();
	}
	
	
	/**
	 * Algorithms that utilize {@link NearestNeighborHeapSearch} 
	 * algorithms for mutual reachability
	 * @author Taylor G Smith
	 */
	abstract class HeapSearchAlgorithm extends HDBSCANLinkageTree {
		int leafSize;
		
		HeapSearchAlgorithm(int leafSize, Collection<Class<? extends GeometricallySeparable>> clzs) {
			super();
			
			this.leafSize = leafSize;
			Class<? extends GeometricallySeparable> clz = metric.getClass();
			if( !clzs.contains(clz) ) {
				warn(clz+" is not a valid distance metric for "+getTreeName()+"Trees. "
						+ "Valid metrics: " + clzs);
				warn("falling back to default metric: " + AbstractClusterer.DEF_DIST);
				model.setSeparabilityMetric(AbstractClusterer.DEF_DIST);
			}
		}

		abstract NearestNeighborHeapSearch getTree(double[][] X);
		abstract String getTreeName();
		
		/**
		 * The linkage function to be used for any classes
		 * implementing the {@link Prim} interface.
		 * @param dt
		 * @return
		 */
		final double[][] primTreeLinkageFunction(double[][] dt) {
			final int min_points = FastMath.min(m - 1, minPts);
			
			long strt = System.currentTimeMillis();
			model.info("building " + getTreeName() + " search tree...");
			NearestNeighborHeapSearch tree = getTree(dt);
			model.info("completed NearestNeighborHeapSearch construction in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-strt, false) + 
					System.lineSeparator());
			
			
			// Query for dists to k nearest neighbors
			Neighborhood query = tree.query(dt, min_points, true, true);
			double[][] dists = query.getDistances();
			double[] coreDistances = MatUtils.getColumn(dists, dists[0].length - 1);
			
			double[][] minSpanningTree = LinkageTreeUtils
					.minSpanTreeLinkageCore_cdist(dt, coreDistances, 
						metric, alpha);
			
			return label(MatUtils.sortAscByCol(minSpanningTree, 2));
		}
		
		/**
		 * The linkage function to be used for any classes
		 * implementing the {@link Boruvka} interface.
		 * @param dt
		 * @return
		 */
		final double[][] boruvkaTreeLinkageFunction(double[][] dt) {
			final int min_points = FastMath.min(m - 1, minPts);
			if(leafSize < 3)
				leafSize = 3;

			long strt = System.currentTimeMillis();
			model.info("building " + getTreeName() + " search tree...");
			NearestNeighborHeapSearch tree = getTree(dt);
			model.info("completed NearestNeighborHeapSearch construction in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-strt, false) + 
					System.lineSeparator());
			
			// We can safely cast the metric to DistanceMetric at this point
			final BoruvkaAlgorithm alg = new BoruvkaAlgorithm(tree, min_points, 
					(DistanceMetric)metric, leafSize / 3, approxMinSpanTree, 
					alpha, model);
			
			double[][] minSpanningTree = alg.spanningTree();
			System.out.println(TestSuite.formatter.format(minSpanningTree));
			return label(MatUtils.sortAscByCol(minSpanningTree, 2));
		}
	}
	
	/**
	 * A class for HDBSCAN algorithms that utilize {@link KDTree}
	 * search spaces for segmenting nearest neighbors
	 * @author Taylor G Smith
	 */
	abstract class KDTreeAlgorithm extends HeapSearchAlgorithm {
		KDTreeAlgorithm(int leafSize) {
			super(leafSize, KDTree.VALID_METRICS);
		}
		
		@Override String getTreeName() { return "KD"; }
		@Override final KDTree getTree(double[][] X) {
			// We can safely cast the sep metric as DistanceMetric
			// after the check in the constructor
			return new KDTree(data, leafSize, 
				(DistanceMetric)metric, model);
		}
	}
	
	/**
	 * A class for HDBSCAN algorithms that utilize {@link BallTree}
	 * search spaces for segmenting nearest neighbors
	 * @author Taylor G Smith
	 */
	abstract class BallTreeAlgorithm extends HeapSearchAlgorithm {
		BallTreeAlgorithm(int leafSize) {
			super(leafSize, BallTree.VALID_METRICS);
		}
		
		@Override String getTreeName() { return "Ball"; }
		@Override final BallTree getTree(double[][] X) {
			// We can safely cast the sep metric as DistanceMetric
			// after the check in the constructor
			return new BallTree(data, leafSize, 
				(DistanceMetric)metric, model);
		}
	}
	
	/**
	 * Generic single linkage tree that uses an 
	 * upper triangular distance matrix to compute
	 * mutual reachability
	 * @author Taylor G Smith
	 */
	class GenericTree extends HDBSCANLinkageTree implements ExplicitMutualReachability {
		GenericTree() {
			super();
			
			// The generic implementation requires the computation of an UT dist mat
			final LogTimer s = new LogTimer();
			dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
			info("completed distance matrix computation in " + s.toString());
		}
		
		@Override
		double[][] link() {
			final double[][] mutual_reachability = mutualReachability();
			double[][] min_spanning_tree = LinkageTreeUtils
					.minSpanTreeLinkageCore(mutual_reachability, m);
			
			// Sort edges of the min_spanning_tree by weight
			min_spanning_tree = MatUtils.sortAscByCol(min_spanning_tree, 2);
			return label(min_spanning_tree);
		}
		
		@Override
		public double[][] mutualReachability() { // Tested: passing
			if(null == dist_mat)
				throw new IllegalClusterStateException("dist matrix is null; "
					+ "this only can happen when the model attempts to invoke "
					+ "mutualReachability on a tree without proper initialization "
					+ "or after the model has already been fit.");
			
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
	
	/**
	 * An implementation of HDBSCAN using the {@link Prim} algorithm
	 * and leveraging {@link KDTree} search spaces
	 * @author Taylor G Smith
	 */
	class PrimsKDTree extends KDTreeAlgorithm implements Prim {
		PrimsKDTree(int leafSize) {
			super(leafSize);
		}
		
		@Override
		double[][] link() {
			return primTreeLinkageFunction(dataData);
		}
	}
	
	/**
	 * An implementation of HDBSCAN using the {@link Prim} algorithm
	 * and leveraging {@link BallTree} search spaces
	 * @author Taylor G Smith
	 */
	class PrimsBallTree extends BallTreeAlgorithm implements Prim {
		PrimsBallTree(int leafSize) {
			super(leafSize);
		}

		@Override
		double[][] link() {
			return primTreeLinkageFunction(dataData);
		}
	}
	
	class BoruvkaKDTree extends KDTreeAlgorithm implements Boruvka {
		BoruvkaKDTree(int leafSize) {
			super(leafSize);
		}

		@Override
		double[][] link() {
			return boruvkaTreeLinkageFunction(dataData);
		}
	}
	
	class BoruvkaBallTree extends BallTreeAlgorithm implements Boruvka {
		BoruvkaBallTree(int leafSize) {
			super(leafSize);
		}

		@Override
		double[][] link() {
			return boruvkaTreeLinkageFunction(dataData);
		}
	}
	
	/**
	 * A base class for any unify finder classes
	 * to extend. These should help join nodes and
	 * branches from trees.
	 * @author Taylor G Smith
	 */
	abstract static class UnifiedFinder {
		final int SIZE;
		
		UnifiedFinder(int N) {
			this.SIZE = N;
		}
		
		/**
		 * Wraps the index in a python way (-1 = last index).
		 * Easier and more concise than having lots of references to 
		 * {@link LinkageTreeUtils#wraparoundIdxGet(int, int)}
		 * @param i
		 * @param j
		 * @return
		 */
		static int wrap(int i, int j) {
			return LinkageTreeUtils.wraparoundIdxGet(i, j);
		}
		
		int wrap(int i) {
			return wrap(SIZE, i);
		}
		
		abstract void union(int m, int n);
		abstract int find(int x);
	}
	
	// Tested: passing
	static class TreeUnionFind extends UnifiedFinder {
		int [][] dataArr;
		boolean [] is_component;
		
		public TreeUnionFind(int size) {
			super(size);
			dataArr = new int[size][2];
			
			// First col should be arange to size
			for(int i = 0; i < size; i++)
				dataArr[i][0] = i;
			
			is_component = VecUtils.repBool(true, size);
		}
		
		@Override
		public void union(int x, int y) {
			int x_root = find(x);
			int y_root = find(y);
			
			int x1idx = wrap(x_root);
			int y1idx = wrap(y_root);
			
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
			final int idx = wrap(x);
			if(dataArr[idx][0] != x) {
				dataArr[idx][0] = find(dataArr[idx][0]);
				is_component[idx] = false;
			}
			
			return dataArr[idx][0];
		}
		
		/**
		 * Returns all non-zero indices in is_component
		 * @return
		 */
		int[] components() {
			final HList<Integer> h = new HList<>();
			for(int i = 0; i < is_component.length; i++)
				if(is_component[i])
					h.add(i);
			
			int idx = 0;
			int[] out = new int[h.size()];
			for(Integer i: h)
				out[idx++] = i;
			
			return out;
		}
	}
	
	// Tested: passing
	static class UnionFind extends UnifiedFinder {
		int [] parent, size;
		int nextLabel;
		
		public UnionFind(int N) {
			super(N);
			parent = VecUtils.repInt(-1, 2 * N - 1);
			nextLabel = N;
			
			size = new int[2 * N - 1];
			for(int i = 0; i < size.length; i++)
				size[i] = i >= N ? 0 : 1; // if N == 5 [1,1,1,1,1,0,0,0,0]
		}
		
		int fastFind(int n) {
			int p = n //,tmp
					;
			
			while(parent[wrap(parent.length, n)] != -1)
				n = parent[wrap(parent.length, n)];
			
			// Incredibly enraging to debug
			while(parent[wrap(parent.length, p)] != n) {
				//System.out.println("First: {p:" + p + ", parent[p]:" +parent[wrap(parent.length, p)] +  ", n:" +n+"}");
				
				//tmp = p;
				p = parent[wrap(parent.length, p)];
				parent[wrap(parent.length, p)] = n;
				
				//System.out.println("Second: {p:" + p + ", parent[p]:" +parent[wrap(parent.length, p)] +  ", n:" +n+"}");
				//System.out.println(Arrays.toString(parent));
			}
			
			return n;
		}
		
		@Override
		public int find(int n) {
			while(parent[wrap(parent.length, n)] != -1)
				n = parent[wrap(parent.length, n)];
			return n;
		}
		
		@Override
		public void union(final int m, final int n) {
			int mWrap = wrap(size.length, m);
			int nWrap = wrap(size.length, n);
			
			size[nextLabel] = size[mWrap] + size[nWrap];
			parent[mWrap] = nextLabel;
			parent[nWrap] = nextLabel;
			size[nextLabel] = size[mWrap] + size[nWrap];
			nextLabel++;
			return;
		}
		
		@Override
		public String toString() {
			return "Parent arr: " + Arrays.toString(parent) + "; " +
					"Sizes: " + Arrays.toString(size) + "; " +
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
		unionFind = new TreeUnionFind(maxParent + 1);
		
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
				info("Model fit:");
				final LogTimer timer = new LogTimer();
				
				// Meant to prevent multiple .getData() copy calls
				dataData = data.getData();
				
				// We don't build the dist mat, because only needed for some algos
				//dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
				
				
				// Build the tree
				String msg = "constructing HDBSCAN single linkage dendrogram: ";
				Class<? extends HDBSCANLinkageTree> clz = null;
				switch(algo) {
					case GENERIC:
						clz = GenericTree.class;
						info(msg + clz.getName());
						tree = new GenericTree();
						break;
						
					case PRIMS_KDTREE:
						clz = PrimsKDTree.class;
						info(msg + clz.getName());
						tree = new PrimsKDTree(leafSize);
						break;
						
					case PRIMS_BALLTREE:
						clz = PrimsBallTree.class;
						info(msg + clz.getName());
						tree = new PrimsBallTree(leafSize);
						break;
					/*	
					case BORUVKA_KDTREE:
						clz = BoruvkaKDTree.class;
						info(msg + clz.getName());
						tree = new BoruvkaKDTree(leafSize);
						break;
						
					case BORUVKA_BALLTREE:
						clz = BoruvkaBallTree.class;
						info(msg + clz.getName());
						tree = new BoruvkaBallTree(leafSize);
						break;
					*/	
					default:
						throw new InternalError("illegal algorithm");
				}
				
				LogTimer treeTimer = new LogTimer();
				final double[][] build = tree.link(); // returns the result of the label(..) function
				info("completed tree building in " + treeTimer.toString());
				

				info("converting tree to labels ("+build.length+" x "+build[0].length+")");
				LogTimer labTimer = new LogTimer();
				labels = treeToLabels(dataData, build, min_cluster_size, this);
				
				
				// Wrap up...
				info("completed cluster labeling in " + labTimer.toString());
				
				
				// Count missing
				numNoisey = 0;
				for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
				
				
				int nextLabel = LabelHSetFactory.build(labels).size() - (numNoisey > 0 ? 1 : 0);
				info((numClusters=nextLabel)+" cluster"+(nextLabel!=1?"s":"")+
					" identified, "+numNoisey+" record"+(numNoisey!=1?"s":"")+
						" classified noise");
				
				
				sayBye(timer);
				
				// Clean anything with big overhead..
				dataData = null;
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
	
	/**
	 * Break up the getLabels method 
	 * into numerous smaller ones.
	 * @author Taylor G Smith
	 */
	static class GetLabelUtils {
		/**
		 * Descendingly sort the keys of the map and return
		 * them in order, but eliminate the very smallest key
		 * @param stability
		 * @return
		 */
		protected static <T,P> HList<T> descSortedKeySet(TreeMap<T,P> stability) {
			int ct = 0;
			HList<T> nodeList = new HList<>();
			for(T d: stability.descendingKeySet())
				if(++ct < stability.size()) // exclude the root...
					nodeList.add(d);
			
			return nodeList;
		}
		
		/**
		 * Get tuples where child size is over one
		 * @param tree
		 * @return
		 */
		protected static EntryPair<HList<double[]>, Integer> childSizeGtOneAndMaxChild(HList<QuadTup<Integer, Integer, Double, Integer>> tree) {
			HList<double[]> out = new HList<>();
			int max = Integer.MIN_VALUE;
			
			// [parent, child, lambda, size]
			for(QuadTup<Integer, Integer, Double, Integer> tup: tree) {
				if(tup.four > 1)
					out.add(new double[]{
						tup.one,
						tup.two,
						tup.three,
						tup.four
					});
				else if(tup.four == 1)
					max = FastMath.max(max, tup.two);
			}
			
			return new EntryPair<>(out, max + 1);
		}
		
		protected static TreeMap<Integer, Boolean> initNodeMap(HList<Integer> nodes) {
			TreeMap<Integer, Boolean> out = new TreeMap<>();
			for(Integer i: nodes)
				out.put(i, true);
			return out;
		}
		
		protected static double subTreeStability(HList<double[]> clusterTree, 
				int node, TreeMap<Integer, Double> stability) {
			double sum = 0;
			
			// [parent, child, lambda, size]
			for(double[] d: clusterTree)
				if((int)d[0] == node)
					sum += stability.get((int)d[1]);
			
			return sum;
		}
		
		protected static HList<Integer> breadthFirstSearchFromClusterTree(HList<double[]> tree, Integer bfsRoot) {
			int child, parent;
			HList<Integer> result = new HList<>();
			HList<Integer> toProcess = new HList<Integer>();
			HList<Integer> tmp;
			
			toProcess.add(bfsRoot);
			
			// [parent, child, lambda, size]
			while(toProcess.size() > 0) {
				result.addAll(toProcess);
				
				// python code: 
				// to_process = tree['child'][np.in1d(tree['parent'], to_process)]
				// For all tuples, if the parent is in toProcess, then
				// add the child to the new list
				tmp = new HList<Integer>();
				for(double[] d: tree) {
					parent	= (int)d[0];
					child	= (int)d[1];
					
					if(toProcess.contains(parent))
						tmp.add(child);
				}
				
				toProcess = tmp;
			}
			
			return result;
		}
	}
	
	protected static int[] getLabels(HList<QuadTup<Integer, Integer, Double, Integer>> condensed,
									TreeMap<Integer, Double> stability) {
		
		double subTreeStability;
		HList<Integer> clusters = new HList<Integer>();
		HSet<Integer> clusterSet;
		TreeMap<Integer, Integer> clusterMap = new TreeMap<>(), 
				reverseClusterMap = new TreeMap<>();
		
		// Get descending sorted key set
		HList<Integer> nodeList = GetLabelUtils.descSortedKeySet(stability);
		
		// Get tuples where child size > 1
		EntryPair<HList<double[]>, Integer> entry = GetLabelUtils.childSizeGtOneAndMaxChild(condensed);
		HList<double[]> clusterTree = entry.getKey();
		
		// Map of nodes to whether it's a cluster
		TreeMap<Integer, Boolean> isCluster = GetLabelUtils.initNodeMap(nodeList);
		
		// Get num points
		//int numPoints = entry.getValue();
		
		// Iter over nodes
		for(Integer node: nodeList) {
			subTreeStability = GetLabelUtils.subTreeStability(clusterTree, node, stability);
			
			if(subTreeStability > stability.get(node)) {
				isCluster.put(node, false);
				stability.put(node, subTreeStability);
			} else {
				for(Integer subNode: GetLabelUtils.breadthFirstSearchFromClusterTree(clusterTree, node))
					if(subNode.intValue() != node)
						isCluster.put(subNode, false);
			}
			
		}
		
		// Now add to clusters
		for(Map.Entry<Integer, Boolean> c: isCluster.entrySet())
			if(c.getValue())
				clusters.add(c.getKey());
		clusterSet = new HSet<Integer>(clusters);
		
		// Build cluster map
		int n = 0;
		for(Integer clust: clusterSet) {
			clusterMap.put(clust, n);
			reverseClusterMap.put(n, clust);
			n++;
		}

		return doLabeling(condensed, clusters, clusterMap);
	}
	
	// Tested: passing
	static double[][] label(final double[][] tree) {
		double[][] result;
		int a, aa, b, bb, index;
		final int m = tree.length, n = tree[0].length, N = m + 1;
		double delta;
		
		result = new double[m][n+1];
		UnionFind U = new UnionFind(N);
		
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
		final double[][] hierarchy = LinkageTreeUtils.minSpanTreeLinkageCore(dists, dists.length);
		return label(MatUtils.sortAscByCol(hierarchy, 2));
	}
	
	protected static int[] treeToLabels(final double[][] X, 
			final double[][] single_linkage_tree, final int min_size) {
		return treeToLabels(X, single_linkage_tree, min_size, null);
	}
	
	protected static int[] treeToLabels(final double[][] X, 
			final double[][] single_linkage_tree, final int min_size, Loggable logger) {
		
		final HList<QuadTup<Integer, Integer, Double, Integer>> condensed = 
				LinkageTreeUtils.condenseTree(single_linkage_tree, min_size);
		final TreeMap<Integer, Double> stability = LinkageTreeUtils.computeStability(condensed);
		
		return getLabels(condensed, stability);
	}
}
