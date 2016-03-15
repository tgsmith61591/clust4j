package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.Loggable;
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
				int leafSize, DistanceMetric sep, Loggable logger);
	}
	
	public static enum NeighborsAlgorithm implements TreeBuilder {
		AUTO {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, Loggable logger) {
				int mn = data.getColumnDimension() * data.getRowDimension();
				return mn > GlobalState.ParallelismConf.MIN_ELEMENTS ?
					BALL_TREE.buildTree(data, leafSize, sep, logger) : 
					KD_TREE.buildTree(data, leafSize, sep, logger);
			}
			
		}, 
		
		KD_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, Loggable logger) {
				return new KDTree(data, leafSize, sep, logger);
			}
			
		}, 
		
		BALL_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(AbstractRealMatrix data,
					int leafSize, DistanceMetric sep, Loggable logger) {
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
	
	abstract Neighborhood getNeighbors(AbstractRealMatrix matrix);
}
