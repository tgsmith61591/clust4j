package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.SimilarityMetric;

public abstract class AbstractDensityClusterer extends AbstractClusterer {
	public AbstractDensityClusterer(AbstractRealMatrix data, AbstractClusterer.BaseClustererPlanner planner) {
		super(data, planner);
		
		// Should not use similarity metrics in DBClusterers, DB looks for 
		// neighborhoods not accurately represented via similarity metrics.
		if(getSeparabilityMetric() instanceof SimilarityMetric && verbose)
			warn("density-based clustering algorithms should use distance metrics instead of similarity metrics");
		
	} // End constructor
}
