package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class AbstractHierarchicalClusterer extends AbstractClusterer {
	public AbstractHierarchicalClusterer(
			AbstractRealMatrix data,
			AbstractClusterer.BaseClustererPlanner planner)
	{
		super(data, planner);
	}
}
