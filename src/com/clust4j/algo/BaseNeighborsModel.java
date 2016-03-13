package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

abstract public class BaseNeighborsModel extends AbstractClusterer {
	private static final long serialVersionUID = 1054047329248586585L;
	public static final Algorithm DEF_ALGO = Algorithm.KD_TREE;

	public static enum Algorithm {
		KD_TREE, BALL_TREE;
	}
	
	protected BaseNeighborsModel(AbstractClusterer caller, BaseNeighborsPlanner planner) {
		super(caller, planner);
	}
	
	public BaseNeighborsModel(AbstractRealMatrix data, BaseNeighborsPlanner planner) {
		super(data, planner);
	}

	abstract public static class BaseNeighborsPlanner 
			extends BaseClustererPlanner 
			implements UnsupervisedClassifierPlanner {
		private static final long serialVersionUID = 8356804193088162871L;
		
		abstract public BaseNeighborsPlanner setAlgorithm(Algorithm algo);
		abstract public Algorithm getAlgorithm();
	}
}
