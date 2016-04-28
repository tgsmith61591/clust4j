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

import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.NamedEntity;
import com.clust4j.kernel.CircularKernel;
import com.clust4j.kernel.LogKernel;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.utils.SimpleHeap;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

import static com.clust4j.metrics.scoring.UnsupervisedMetric.SILHOUETTE;

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
 * @see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html">Agglomerative Clustering</a>
 * @see <a href="http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm">Divisive Clustering</a>
 */
final public class HierarchicalAgglomerative extends AbstractPartitionalClusterer implements UnsupervisedClassifier {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7563413590708853735L;
	public static final Linkage DEF_LINKAGE = Linkage.WARD;
	final static HashSet<Class<? extends GeometricallySeparable>> comp_avg_unsupported;
	static {
		comp_avg_unsupported = new HashSet<>();
		comp_avg_unsupported.add(CircularKernel.class);
		comp_avg_unsupported.add(LogKernel.class);
	}
	
	/**
	 * Which {@link Linkage} to use for the clustering algorithm
	 */
	final Linkage linkage;
	
	interface LinkageTreeBuilder extends MetricValidator {
		public HierarchicalDendrogram buildTree(HierarchicalAgglomerative h);
	}
	
	/**
	 * The linkages for agglomerative clustering. 
	 * @author Taylor G Smith
	 */
	public enum Linkage implements java.io.Serializable, LinkageTreeBuilder {
		AVERAGE {
			@Override
			public AverageLinkageTree buildTree(HierarchicalAgglomerative h) {
				return h.new AverageLinkageTree();
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable geo) {
				return !comp_avg_unsupported.contains(geo.getClass());
			}
		}, 
		
		COMPLETE {
			@Override
			public CompleteLinkageTree buildTree(HierarchicalAgglomerative h) {
				return h.new CompleteLinkageTree();
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable geo) {
				return !comp_avg_unsupported.contains(geo.getClass());
			}
		}, 
		
		WARD {
			@Override
			public WardTree buildTree(HierarchicalAgglomerative h) {
				return h.new WardTree();
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable geo) {
				return geo.equals(Distance.EUCLIDEAN);
			}
		};
	}
	
	
	@Override
	final public boolean isValidMetric(GeometricallySeparable geo) {
		return this.linkage.isValidMetric(geo);
	}
	
	
	
	
	/**
	 * The number of rows in the matrix
	 */
	final private int m;
	
	
	/**
	 * The labels for the clusters
	 */
	volatile private int[] labels = null;
	/**
	 * The flattened distance vector
	 */
	volatile private EfficientDistanceMatrix dist_vec = null;
	volatile HierarchicalDendrogram tree = null;
	/** 
	 * Volatile because if null will later change during build
	 */
	volatile private int num_clusters;
	
	
	
	
	
	
	protected HierarchicalAgglomerative(RealMatrix data) {
		this(data, new HierarchicalAgglomerativeParameters());
	}

	protected HierarchicalAgglomerative(RealMatrix data, 
			HierarchicalAgglomerativeParameters planner) {
		super(data, planner, planner.getNumClusters());
		this.linkage = planner.getLinkage();
		
		if(!isValidMetric(this.dist_metric)) {
			warn(this.dist_metric.getName() + " is invalid for " + this.linkage + 
				". Falling back to default Euclidean dist");
			setSeparabilityMetric(DEF_DIST);
		}
		
		this.m = data.getRowDimension();
		this.num_clusters = super.k;
		
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Linkage","Allow Par.","Num. Clusters"
			}, new Object[]{
				data.getRowDimension(),data.getColumnDimension(),
				getSeparabilityMetric(),linkage,
				parallel,
				num_clusters
			});
	}
	
	
	
	
	
	
	
	/**
	 * Computes a flattened upper triangular distance matrix in a much more space efficient manner,
	 * however traversing it requires intermittent calculations using {@link #navigate(int, int, int)}
	 * @author Taylor G Smith
	 */
	protected static class EfficientDistanceMatrix implements java.io.Serializable {
		private static final long serialVersionUID = -7329893729526766664L;
		final protected double[] dists;
		
		EfficientDistanceMatrix(final RealMatrix data, GeometricallySeparable dist, boolean partial) {
			this.dists = build(data.getData(), dist, partial);
		}
		
		/**
		 * Copy constructor
		 */
		/*// not needed right now...
		private EfficientDistanceMatrix(EfficientDistanceMatrix other) {
			this.dists = VecUtils.copy(other.dists);
		}
		*/
		
		/**
		 * Computes a flattened upper triangular distance matrix in a much more space efficient manner,
		 * however traversing it requires intermittent calculations using {@link #navigateFlattenedMatrix(double[], int, int, int)}
		 * @param data
		 * @param dist
		 * @param partial -- use the partial distance?
		 * @return a flattened distance vector
		 */
		static double[] build(final double[][] data, GeometricallySeparable dist, boolean partial) {
			final int m = data.length;
			final int s = m*(m-1)/2; // The shape of the flattened upper triangular matrix (m choose 2)
			final double[] vec = new double[s];
			for(int i = 0, r = 0; i < m - 1; i++)
				for(int j = i + 1; j < m; j++, r++)
					vec[r] = partial ? dist.getPartialDistance(data[i], data[j]) : 
						dist.getDistance(data[i], data[j]);
			
			return vec;
		}
		
		/**
		 * For a flattened upper triangular matrix...
		 * 
		 * <p>
		 * Original:
		 * <p>
		 * <table>
		 * <tr><td>0 </td><td>1 </td><td>2 </td><td>3</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>1 </td><td>2</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>0 </td><td>1</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>0 </td><td>0</td></tr>
		 * </table>
		 * 
		 * <p>
		 * Flattened:
		 * <p>
		 * &lt;1 2 3 1 2 1&gt;
		 * 
		 * <p>
		 * ...and the parameters <tt>m</tt>, the original row dimension,
		 * <tt>i</tt> and <tt>j</tt>, will identify the corresponding index
		 * in the flattened vector such that mat[0][3] corresponds to vec[2];
		 * this method, then, would return 2 (the index in the vector 
		 * corresponding to mat[0][3]) in this case.
		 * 
		 * @param m
		 * @param i
		 * @param j
		 * @return the corresponding vector index
		 */
		static int getIndexFromFlattenedVec(final int m, final int i, final int j) {
			if(i < j)
				return m * i - (i * (i + 1) / 2) + (j - i - 1);
			else if(i > j)
				return m * j - (j * (j + 1) / 2) + (i - j - 1);
			throw new IllegalArgumentException(i+", "+j+"; i should not equal j");
		}
		
		/**
		 * For a flattened upper triangular matrix...
		 * 
		 * <p>
		 * Original:
		 * <p>
		 * <table>
		 * <tr><td>0 </td><td>1 </td><td>2 </td><td>3</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>1 </td><td>2</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>0 </td><td>1</td></tr>
		 * <tr><td>0 </td><td>0 </td><td>0 </td><td>0</td></tr>
		 * </table>
		 * 
		 * <p>
		 * Flattened:
		 * <p>
		 * &lt;1 2 3 1 2 1&gt;
		 * 
		 * <p>
		 * ...and the parameters <tt>m</tt>, the original row dimension,
		 * <tt>i</tt> and <tt>j</tt>, will identify the corresponding value
		 * in the flattened vector such that mat[0][3] corresponds to vec[2];
		 * this method, then, would return 3, the value at index 2, in this case.
		 * 
		 * @param m
		 * @param i
		 * @param j
		 * @return the corresponding vector index
		 */
		double navigate(final int m, final int i, final int j) {
			return dists[getIndexFromFlattenedVec(m,i,j)];
		}
	}
	
	abstract class HierarchicalDendrogram implements java.io.Serializable, NamedEntity {
		private static final long serialVersionUID = 5295537901834851676L;
		public final HierarchicalAgglomerative ref;
		public final GeometricallySeparable dist;
		
		HierarchicalDendrogram() {
			ref = HierarchicalAgglomerative.this;
			dist = ref.getSeparabilityMetric();
			
			if(null == dist_vec) // why would this happen?
				dist_vec = new EfficientDistanceMatrix(data, dist, true);
		}
		
		double[][] linkage() {
			// Perform the linkage logic in the tree
			//EfficientDistanceMatrix y = dist_vec.copy(); // Copy the dist_vec
			
			double[][] Z = new double[m - 1][4];  // Holding matrix
			link(dist_vec, Z, m); // Immutabily change Z
			
			// Final linkage tree out...
			return MatUtils.getColumns(Z, new int[]{0,1});
		}
		
		private void link(final EfficientDistanceMatrix dists, final double[][] Z, final int n) {
			int i, j, k, x = -1, y = -1, i_start, nx, ny, ni, id_x, id_y, id_i, c_idx;
			double current_min;
			
			// Inter cluster dists
			EfficientDistanceMatrix D = dists; //VecUtils.copy(dists);
			
			// Map the indices to node ids
			ref.info("initializing node mappings ("+getClass().getName().split("\\$")[1]+")");
			int[] id_map = new int[n];
			for(i = 0; i < n; i++) 
				id_map[i] = i;
			
			LogTimer link_timer = new LogTimer(), iterTimer;
			int incrementor = n/10, pct = 1;
			for(k = 0; k < n - 1; k++) {
				if(incrementor>0 && k%incrementor == 0)
					ref.info("node mapping progress - " + 10*pct++ + "%. Total link time: "+
						link_timer.toString()+"");
				
				// get two closest x, y
				current_min = Double.POSITIVE_INFINITY;
				
				iterTimer = new LogTimer();
				for(i = 0; i < n - 1; i++) {
					if(id_map[i] == -1)
						continue;
					
					
					i_start = EfficientDistanceMatrix.getIndexFromFlattenedVec(n, i, i + 1);
					for(j = 0; j < n - i - 1; j++) {
						if(D.dists[i_start + j] < current_min) {
							current_min = D.dists[i_start + j];
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
				int cont = 0;
				for(i = 0; i < n; i++) {
					id_i = id_map[i];
					if(id_i == -1 || id_i == n + k) {
						cont++;
						continue;
					}
					
					ni = id_i < n ? 1 : (int)Z[id_i - n][3];
					c_idx = EfficientDistanceMatrix.getIndexFromFlattenedVec(n, i, y);
					D.dists[c_idx] = getDist(D.navigate(n, i, x), D.dists[c_idx], current_min, nx, ny, ni);
					
					if(i < x)
						D.dists[EfficientDistanceMatrix.getIndexFromFlattenedVec(n,i,x)] = Double.POSITIVE_INFINITY;
				}
				
				fitSummary.add(new Object[]{
					k,current_min,cont,iterTimer.formatTime(),
					link_timer.formatTime(),link_timer.wallMsg()
				});
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
		
		@Override
		public String getName() {
			return "Ward Tree";
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
		
		@Override
		public String getName() {
			return "Avg Linkage Tree";
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
		
		@Override
		public String getName() {
			return "Complete Linkage Tree";
		}
	}
	
	
	
	@Override
	public String getName() {
		return "Agglomerative";
	}

	public Linkage getLinkage() {
		return linkage;
	}

	@Override
	protected HierarchicalAgglomerative fit() {
		synchronized(fitLock) {
			if(null != labels) // already fit
				return this;
			
			final LogTimer timer = new LogTimer();
			labels = new int[m];
			
			/*
			 * Corner case: k = 1 (due to singularity?)
			 */
			if(1 == k) {
				this.fitSummary.add(new Object[]{
					0,0,Double.NaN,timer.formatTime(),timer.formatTime(),timer.wallMsg()
				});
				
				warn("converged immediately due to " + (this.singular_value ? 
						"singular nature of input matrix" : "k = 1"));
				sayBye(timer);
				return this;
			}
			
			dist_vec = new EfficientDistanceMatrix(data, getSeparabilityMetric(), true);
			
			// Log info...
			info("computed distance matrix in " + timer.toString());
			
			
			// Get the tree class for logging...
			LogTimer treeTimer = new LogTimer();
			this.tree = this.linkage.buildTree(this);
			
			// Tree build
			info("constructed " + tree.getName() + " HierarchicalDendrogram in " + treeTimer.toString());
			double[][] children = tree.linkage();
			
			
			
			// Cut the tree
			labels = hcCut(num_clusters, children, m);
			labels = new SafeLabelEncoder(labels).fit().getEncodedLabels();
			
			
			sayBye(timer);
			dist_vec = null;
			return this;
		}
		
	} // End train
	
	static int[] hcCut(final int n_clusters, final double[][] children, final int n_leaves) {
		/*
		 * Leave children as a double[][] despite it
		 * being ints. This will allow VecUtils to operate
		 */
		
		if(n_clusters > n_leaves)
			throw new InternalError(n_clusters + " > " + n_leaves);
		
		// Init nodes
		SimpleHeap<Integer> nodes = new SimpleHeap<>(-((int)VecUtils.max(children[children.length-1]) + 1));

		
		for(int i = 0; i < n_clusters - 1; i++) {
			int inner_idx = -nodes.get(0) - n_leaves;
			if(inner_idx < 0)
				inner_idx = children.length + inner_idx;
			
			double[] these_children = children[inner_idx];
			nodes.push(-((int)these_children[0]));
			nodes.pushPop(-((int)these_children[1]));
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
		if(node < leaves)
			return new Integer[]{node};

		final SimpleHeap<Integer> ind = new SimpleHeap<>(node);
		final ArrayList<Integer> descendent = new ArrayList<>();
		int i, n_indices = 1;
		
		while(n_indices > 0) {
			i = ind.popInPlace();
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
	
	@Override
	public int[] getLabels() {
		return super.handleLabelCopy(labels);
	}
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AGGLOMERATIVE;
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Link Iter. #","Iter. Min","Continues","Iter. Time","Total Time","Wall"
		};
	}
	
	/** {@inheritDoc} */
	@Override
	public double indexAffinityScore(int[] labels) {
		// Propagates ModelNotFitException
		return SupervisedMetric.INDEX_AFFINITY.evaluate(labels, getLabels());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		// Propagates ModelNotFitException
		return SILHOUETTE.evaluate(this, getLabels());
	}
	
	/** {@inheritDoc} */
	@Override
	public int[] predict(RealMatrix newData) {
		final int[] fit_labels = getLabels(); // throws the MNF exception if not fit
		final int numSamples = newData.getRowDimension(), n = newData.getColumnDimension();
		
		// Make sure matches dimensionally
		if(n != this.data.getColumnDimension())
			throw new DimensionMismatchException(n, data.getColumnDimension());
		
		/*
		 * There's no great way to predict on a hierarchical
		 * algorithm, so we'll treat this like a CentroidLearner,
		 * create centroids from the k clusters formed, then
		 * predict via the CentroidUtils. This works because
		 * Hierarchical is not a NoiseyClusterer
		 */
		
		// CORNER CASE: num_clusters == 1, return only label (0)
		if(1 == num_clusters)
			return VecUtils.repInt(fit_labels[0], numSamples);
		
		return new NearestCentroidParameters()
			.setMetric(this.dist_metric) // if it fails, falls back to default Euclidean...
			.setVerbose(false) // just to be sure in case default ever changes...
			.fitNewModel(this.getData(), fit_labels)
		.predict(newData);
	}
}
