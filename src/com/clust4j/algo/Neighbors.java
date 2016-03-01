package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.DistanceMetric;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.ModelNotFitException;

abstract public class Neighbors extends BaseNeighborsModel {
	private static final long serialVersionUID = 527294226423813447L;
	public static final int DEF_LEAF_SIZE = 30;
	public static final int DEF_K = 5;
	public static final double DEF_RADIUS = 5.0;
	public final static boolean DUAL_TREE_SEARCH = false;
	public final static boolean SORT = true;
	
	Integer kNeighbors = null;
	Double radius = null;
	final NearestNeighborHeapSearch tree;
	final boolean radiusMode;
	final int leafSize, m;
	final double[][] fit_X;

	volatile Neighborhood res;
	
	
	public Neighbors(AbstractRealMatrix data, NeighborsPlanner planner) {
		super(data, planner);
		
		this.kNeighbors = planner.getK();
		this.radius = planner.getRadius();
		this.leafSize = planner.getLeafSize();
		
		radiusMode = null != radius;
		
		
		if(!(planner.getSep() instanceof DistanceMetric)) {
			warn(planner.getSep() + " not a valid metric. Falling back to default: " + DEF_DIST);
			super.setSeparabilityMetric(DEF_DIST);
		}
		
		if(leafSize < 1)
			throw new IllegalArgumentException("leafsize must be positive");
		
		
		GeometricallySeparable sep = this.getSeparabilityMetric();
		switch(planner.getAlgorithm()) {
			// We can cast to DistanceMetric at this point
			case KD_TREE:	
				tree = new KDTree(data,  leafSize, (DistanceMetric)sep, this);
				break;
			case BALL_TREE:	
				tree = new BallTree(data,leafSize, (DistanceMetric)sep, this);
				break;
			default: 
				/* Can't test for coverage; throws NPE in switch declaration.
				 * This IAE should never EVER be thrown */
				throw new IllegalArgumentException("unknown algorithm");
		}
		
		// Get the data ref from the tree
		fit_X = tree.getData();
		this.m = fit_X.length;
		
		meta(radiusMode?("radius="+radius):("k="+kNeighbors));
		meta("leafSize="+leafSize);
		meta("algorithm="+planner.getAlgorithm());
	}
	
	
	
	abstract static class NeighborsPlanner extends BaseNeighborsPlanner {
		abstract public Integer getK();
		abstract public int getLeafSize();
		abstract public Double getRadius();
	}

	@Override
	final public Algo getLoggerTag() {
		return Algo.NEAREST;
	}
	
	public Neighborhood getNeighbors() {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		
		return res.copy();
	}
	
	abstract Neighborhood getNeighbors(AbstractRealMatrix matrix);
}
