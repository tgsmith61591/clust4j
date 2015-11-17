package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.AgglomerativeClusterTree;

public class AgglomerativeClusterer extends AbstractHierarchicalClusterer {
	private AgglomerativeClusterTree tree = null;

	public AgglomerativeClusterer(AbstractRealMatrix data, 
			AbstractHierarchicalClusterer.BaseHierarchicalPlanner planner) {
		super(data, planner);
	}

	@Override
	public String getName() {
		return "Agglomerative";
	}
	
	public AgglomerativeClusterTree getTree() {
		return tree;
	}

	@Override
	public boolean isTrained() {
		return isTrained;
	}

	@Override
	public void train() {
		if(isTrained)
			return;
		
		tree = AgglomerativeClusterTree.build(data.getData(), getDistanceMetric(), false);
		isTrained = true;
	}
}
