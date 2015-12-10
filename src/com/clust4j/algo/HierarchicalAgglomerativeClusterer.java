package com.clust4j.algo;

import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.IllegalClusterStateException;

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
public class HierarchicalAgglomerativeClusterer extends AbstractPartitionalClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7563413590708853735L;
	public static final Linkage DEF_LINKAGE = Linkage.WARD;
	
	/**
	 * Which {@link Linkage} to use for the clustering algorithm
	 */
	private final Linkage linkage;
	/**
	 * The number of rows in the matrix
	 */
	private final int m, n;
	
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
	
	
	
	
	
	
	public HierarchicalAgglomerativeClusterer(AbstractRealMatrix data, int k) {
		this(data, new HierarchicalPlanner(k));
	}

	public HierarchicalAgglomerativeClusterer(AbstractRealMatrix data, 
			HierarchicalPlanner planner) {
		super(data, planner, planner.k);
		
		this.linkage = planner.linkage;
		if(null == linkage) {
			String e = "null linkage passed to planner";
			if(verbose) error(e);
			throw new IllegalClusterStateException(e);
		}
		
		this.m = data.getRowDimension();
		this.n = data.getColumnDimension();
		if(verbose) meta("Linkage="+linkage);
		if(verbose) meta("Num clusters="+k);
		if(verbose) warn(getName()+" clustering has a runtime of O(N^2)");
	}
	
	
	
	
	
	public static class HierarchicalPlanner extends AbstractClusterer.BaseClustererPlanner {
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private Linkage linkage = DEF_LINKAGE;
		private boolean verbose = DEF_VERBOSE;
		private int k;
		
		
		public HierarchicalPlanner(int k){ this.k = k; }
		public HierarchicalPlanner(int k, Linkage linkage) {
			this(k);
			this.linkage = linkage;
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
	}
	
	
	
	
	
	abstract public class HierarchicalDendrogram implements java.io.Serializable {
		private static final long serialVersionUID = 5295537901834851676L;
		
		public HierarchicalDendrogram(){
			if(null == dist_mat)
				dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
		}
	}
	
	public class WardTree extends HierarchicalDendrogram {
		private static final long serialVersionUID = -2336170779406847047L;
		
		public WardTree(){ super(); }
		
	}
	
	abstract public class LinkageTree extends HierarchicalDendrogram {
		private static final long serialVersionUID = -252115690411913842L;
		public LinkageTree(){ super(); }
		
		abstract public TreeMap<Integer, Double> merge(TreeMap<Integer, Double> a, 
			TreeMap<Integer, Double> b, boolean[] mask, 
			float n_a, float n_b);
	}
	
	public class AverageLinkageTree extends LinkageTree {
		private static final long serialVersionUID = 5891407873391751152L;

		public AverageLinkageTree(){ super(); }

		@Override public TreeMap<Integer, Double> merge(TreeMap<Integer, 
				Double> a, TreeMap<Integer, Double> b, 
				boolean[] mask, float n_a, float n_b) {
			
			TreeMap<Integer, Double> out = new TreeMap<Integer, Double>();
			Iterator<Map.Entry<Integer, Double>> a_it = a.entrySet().iterator();
			Iterator<Map.Entry<Integer, Double>> a_end= a.descendingMap().entrySet().iterator();
			
			Map.Entry<Integer, Double> entry;
			Integer key = a.firstKey(), last_key = a.lastKey();
			Double value;
			
			// Copy a into out if not prevented by mask
			while(!key.equals(last_key)) {
				entry = a_it.next();
				key = entry.getKey();
				if(mask[key])
					out.put(key, entry.getValue());
			}
			
			// Merge b into out
			Iterator<Map.Entry<Integer, Double>> out_it = out.entrySet().iterator();
			Iterator<Map.Entry<Integer, Double>> out_end= out.descendingMap().entrySet().iterator();
			Iterator<Map.Entry<Integer, Double>> b_it = b.entrySet().iterator();
			Iterator<Map.Entry<Integer, Double>> b_end =b.descendingMap().entrySet().iterator();
			
			key = b.firstKey();
			last_key = b.lastKey();
			
			while(!key.equals(last_key)) {
				entry = b_it.next();
				key = entry.getKey();
				value = entry.getValue();
				if(mask[key]) {
					// TODO
				}
			}
			
			return out;
		}
	}
	
	public class CompleteLinkageTree extends LinkageTree {
		private static final long serialVersionUID = 7407993870975009576L;
		
		public CompleteLinkageTree(){ super(); }
		
		@Override public TreeMap<Integer, Double> merge(TreeMap<Integer, 
				Double> a, TreeMap<Integer, Double> b, 
				boolean[] mask, float n_a, float n_b) {
			// TODO Auto-generated method stub
			return null;
		}
	}

	
	
	
	
	@Override
	public String getName() {
		return "Agglomerative";
	}

	@Override
	public HierarchicalAgglomerativeClusterer fit() {
		synchronized(this) { // synch because alters internal structs
			
			if(null != labels) // Then we've already fit this...
				return this;
			
			
			final long start = System.currentTimeMillis();
			labels = new int[m];
			dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
			
			// Log info...
			if(verbose)
				info("calculated " + 
					m + " x " + m + 
					" distance matrix in " + 
					LogTimeFormatter.millis( System.currentTimeMillis()-start , false));
			if(verbose) info("constructing HierarchicalDendrogram");
			
			
			// Build the tree
			switch(linkage) {
				case WARD:
					tree = new WardTree();
					break;
				case AVERAGE:
					tree = new AverageLinkageTree();
					break;
				case COMPLETE:
					tree = new CompleteLinkageTree();
					break;
				default:
					throw new InternalError("illegal linkage");
			}
			
			// TODO
			
			
			
			if(verbose) 
				info("model " + getKey() + " completed in " + 
						LogTimeFormatter.millis(System.currentTimeMillis()-start, false) +
						System.lineSeparator());
			
			dist_mat = null;
			return this;
			
		} // End synch
	} // End train
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AGGLOMERATIVE;
	}
}
