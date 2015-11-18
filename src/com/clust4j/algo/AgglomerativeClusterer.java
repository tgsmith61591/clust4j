package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Linkage;
import com.clust4j.utils.SingleLinkageACTree;

public class AgglomerativeClusterer extends AbstractHierarchicalClusterer {
	private SingleLinkageACTree tree = null;
	
	public AgglomerativeClusterer(AbstractRealMatrix data) {
		super(data, new AbstractHierarchicalClusterer.BaseHierarchicalPlanner());
	}

	public AgglomerativeClusterer(AbstractRealMatrix data, 
			AbstractHierarchicalClusterer.BaseHierarchicalPlanner planner) {
		super(data, planner);
	}

	@Override
	public String getName() {
		return "Agglomerative";
	}
	
	public SingleLinkageACTree getTree() {
		return tree;
	}

	@Override
	public boolean isTrained() {
		return isTrained;
	}
	
	@Override
	public String toString() {
		if(null == tree) return super.toString();
		return super.toString() + ": " + tree.toString();
	}

	@Override
	public void train() {
		if(isTrained)
			return;
		
		buildTree(linkage);
		isTrained = true;
	}
	
	private void buildTree(Linkage link) {
		if(null == link) {
			String e = "null linkage passed to planner";
			if(verbose)
				error(e);
			throw new IllegalArgumentException(e);
		}
		
		switch(link) {
			case SINGLE:
				if(verbose) info("single linkage selected -- building SingleLinkageACTree");
				tree = SingleLinkageACTree
						.build(data.getData(), 
								getDistanceMetric(), 
								false, this);
				break;
			default:
				if(verbose) error("unimplemented linkage method");
				throw new IllegalArgumentException("unimplemented linkage method");
		}
	}

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AGGLOM_;
	}
}
