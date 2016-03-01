package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class AbstractPartitionalClusterer extends AbstractClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8489725366968682469L;
	final protected int k;
	
	public AbstractPartitionalClusterer(
			AbstractRealMatrix data, 
			AbstractClusterer.BaseClustererPlanner planner,
			final int k) 
	{
		super(data, planner);
		
		if(k < 1)
			throw new IllegalArgumentException("k must exceed 0");
		if(k > data.getRowDimension())
			throw new IllegalArgumentException("k exceeds number of records");
		
		this.k = k;
	} // End constructor
	
	public int getK() {
		return k;
	}
}
