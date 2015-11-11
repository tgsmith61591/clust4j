package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class AbstractDensityClusterer extends AbstractClusterer {
	public AbstractDensityClusterer(
			AbstractRealMatrix data, 
			AbstractClusterer.BaseClustererPlanner planner) 
		{
			super(data, planner);
		} // End constructor
}
