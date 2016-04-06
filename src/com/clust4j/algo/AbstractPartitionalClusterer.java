package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class AbstractPartitionalClusterer extends AbstractClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8489725366968682469L;
	/**
	 * The number of clusters to find. This field is not final, as in
	 * some corner cases, the algorithm will modify k for convergence.
	 */
	protected int k;
	
	public AbstractPartitionalClusterer(
			AbstractRealMatrix data, 
			AbstractClusterer.BaseClustererPlanner planner,
			final int k) 
	{
		super(data, planner);
		
		if(k < 1)
			error(new IllegalArgumentException("k must exceed 0"));
		if(k > data.getRowDimension())
			error(new IllegalArgumentException("k exceeds number of records"));
		
		this.k = this.singular_value ? 1 : k;
		if(this.singular_value && k!=1) {
			warn("coerced k to 1 due to equality of all elements in input matrix");
		}
	} // End constructor
	
	public int getK() {
		return k;
	}
}
