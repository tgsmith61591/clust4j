package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.metrics.pairwise.DistanceMetric;

abstract public class BaseNeighborsModel extends AbstractClusterer {
	private static final long serialVersionUID = 1054047329248586585L;
	
	public static final NeighborsAlgorithm DEF_ALGO = NeighborsAlgorithm.AUTO;
	public static final int DEF_LEAF_SIZE = 30;
	public static final int DEF_K = 5;
	public static final double DEF_RADIUS = 5.0;
	public final static boolean DUAL_TREE_SEARCH = false;
	public final static boolean SORT = true;

	protected Integer kNeighbors = null;
	protected Double radius = null;
	protected boolean radiusMode;
	protected int leafSize, m;
	protected double[][] fit_X;
	protected NearestNeighborHeapSearch tree;
	protected NeighborsAlgorithm alg;
	
	/** Resultant neighborhood from fit method */
	volatile Neighborhood res;

	interface TreeBuilder extends java.io.Serializable {
		public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data, 
				int leafSize, DistanceMetric sep, BaseNeighborsModel logger);
	}
	
	public static enum NeighborsAlgorithm implements TreeBuilder {
		AUTO {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, BaseNeighborsModel logger) {
				int mn = data.getColumnDimension() * data.getRowDimension();
				logger.alg = mn > GlobalState.ParallelismConf.MIN_ELEMENTS ?
					BALL_TREE : KD_TREE;
				
				return logger.alg.buildTree(data, leafSize, sep, logger);
			}
			
		}, 
		
		KD_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, BaseNeighborsModel logger) {
				return new KDTree(data, leafSize, sep, logger);
			}
			
		}, 
		
		BALL_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, BaseNeighborsModel logger) {
				return new BallTree(data, leafSize, sep, logger);
			}
			
		};
	}
	
	protected BaseNeighborsModel(AbstractClusterer caller, BaseNeighborsPlanner planner) {
		super(caller, planner);
		init(planner);
	}
	
	protected BaseNeighborsModel(AbstractRealMatrix data, BaseNeighborsPlanner planner, boolean as_is) {
		super(data, planner, as_is);
		init(planner);
	}
	
	public BaseNeighborsModel(AbstractRealMatrix data, BaseNeighborsPlanner planner) {
		super(data, planner);
		init(planner);
	}
	
	final private void init(BaseNeighborsPlanner planner) {
		this.kNeighbors = planner.getK();
		this.radius = planner.getRadius();
		this.leafSize = planner.getLeafSize();
		
		radiusMode = null != radius;
		
		
		if(!(planner.getSep() instanceof DistanceMetric)) {
			warn(planner.getSep() + " not a valid metric for neighbors models. "
				+ "Falling back to default: " + DEF_DIST);
			super.setSeparabilityMetric(DEF_DIST);
		}
		
		if(leafSize < 1)
			throw new IllegalArgumentException("leafsize must be positive");
		
		
		DistanceMetric sep = (DistanceMetric)this.getSeparabilityMetric();
		this.alg = planner.getAlgorithm();
		this.tree = this.alg.buildTree(this.data, this.leafSize, sep, this);
		
		// Get the data ref from the tree
		fit_X = tree.getData();
		this.m = fit_X.length;
	}

	abstract public static class BaseNeighborsPlanner 
			extends BaseClustererPlanner 
			implements UnsupervisedClassifierPlanner {
		private static final long serialVersionUID = 8356804193088162871L;
		
		abstract public BaseNeighborsPlanner setAlgorithm(NeighborsAlgorithm algo);
		abstract public NeighborsAlgorithm getAlgorithm();
		abstract public Integer getK();
		abstract public int getLeafSize();
		abstract public Double getRadius();
	}
	
	public Neighborhood getNeighbors() {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		return res.copy();
	}
	
	/**
	 * The query methods end up adding one to the front...
	 * @param existing
	 * @return
	 */
	protected static Neighborhood trimFirst(Neighborhood existing) {
		final double[][] d_in = existing.getDistances();
		final int[][] i_in = existing.getIndices();
		
		final double[][] d_out = new double[d_in.length][];
		final int[][] i_out = new int[i_in.length][];
		
		int l;
		for(int i = 0; i < d_in.length; i++) {
			l = d_in[i].length;
			
			d_out[i] = new double[l - 1];
			i_out[i] = new int[l - 1];
			
			for(int j = 1, k = 0; j < l; j++, k++) {
				d_out[i][k] = d_in[i][j];
				i_out[i][k] = i_in[i][j];
			}
		}
		
		return new Neighborhood(d_out, i_out);
	}
	
	/**
	 * A class to query the tree for neighborhoods in parallel
	 * @author Taylor G Smith
	 */
	abstract static class ParallelNeighborhoodSearch extends ParallelChunkingTask<Neighborhood> {
		private static final long serialVersionUID = -1600812794470325448L;
		
		final BaseNeighborsModel model;
		final double[][] distances;
		final int[][] indices;
		final int lo;
		final int hi;

		public ParallelNeighborhoodSearch(double[][] X, BaseNeighborsModel model) {
			super(X); // this auto-chunks the data
			
			this.model = model;
			this.lo = 0;
			this.hi = strategy.getNumChunks(X);
			
			/*
			 * First get the length...
			 */
			int length = 0;
			for(Chunk c: this.chunks)
				length += c.size();
			
			this.distances = new double[length][];
			this.indices = new int[length][];
		}
		
		public ParallelNeighborhoodSearch(ParallelNeighborhoodSearch task, int lo, int hi) {
			super(task);
			
			this.model = task.model;
			this.lo = lo;
			this.hi = hi;
			this.distances = task.distances;
			this.indices = task.indices;
		}

		@Override
		public Neighborhood reduce(Chunk chunk) {
			Neighborhood n = query(model.tree, chunk.get());
			
			// assign to low index, since that's how we retrieved the chunk...
			final int start = chunk.start , end = start + chunk.size();
			double[][] d = n.getDistances();
			int[][] i = n.getIndices();
			
			// Set the distances and indices in place...
			for(int j = start, idx = 0; j < end; j++, idx++) {
				this.distances[j] = d[idx];
				this.indices[j] = i[idx];
			}
			
			return n;
		}

		@Override
		protected Neighborhood compute() {
			if(hi - lo <= 1) { // generally should equal one...
				return reduce(chunks.get(lo));
			} else {
				int mid = this.lo + (this.hi - this.lo) / 2;
				ParallelNeighborhoodSearch left  = newInstance(this, this.lo, mid);
				ParallelNeighborhoodSearch right = newInstance(this, mid, this.hi);
				
				left.fork();
	            right.compute();
	            left.join();

	            return new Neighborhood(distances, indices);
			}
		}
		
		abstract ParallelNeighborhoodSearch newInstance(ParallelNeighborhoodSearch p, int lo, int hi);
		abstract Neighborhood query(NearestNeighborHeapSearch tree, double[][] X);
	}
	
	
	abstract Neighborhood getNeighbors(AbstractRealMatrix matrix);
}
