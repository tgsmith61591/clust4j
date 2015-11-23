package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Linkage;
import com.clust4j.utils.HierarchicalClusterTree;
import com.clust4j.utils.SingleLinkageAgglomerativeFactory;

/**
 * Agglomerative clustering is a hierarchical clustering process in
 * which each input record initially is mapped to its own cluster.
 * Progressively, each cluster is merged by locating the least dissimilar 
 * clusters in a M x M distance matrix, merging them, removing the corresponding
 * rows and columns from the distance matrix and adding a new row/column vector
 * of distances corresponding to the new cluster until there is one cluster.
 * <p>
 * Agglomerative clustering does <i>not</i> scale well to large data, performing
 * at O(n<sup>2</sup>) computationally, yet it outperforms its cousin, Divisive Clustering 
 * (DIANA), which performs at O(2<sup>n</sup>).
 * 
 * @author Taylor G Smith
 * 
 * @see com.clust4j.utils.SingleLinkageAgglomerativeFactory
 * @see com.clust4j.utils.HierarchicalClusterTree
 * @see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html">Agglomerative Clustering</a>
 * @see <a href="http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm">Divisive Clustering</a>
 */
public class AgglomerativeClusterer extends AbstractHierarchicalClusterer {
	private HierarchicalClusterTree tree = null;
	
	public AgglomerativeClusterer(AbstractRealMatrix data) {
		super(data, new AbstractHierarchicalClusterer.BaseHierarchicalPlanner());
	}

	public AgglomerativeClusterer(AbstractRealMatrix data, 
			AbstractHierarchicalClusterer.BaseHierarchicalPlanner planner) {
		super(data, planner);
		
		if(verbose) info("Linkage="+linkage);
	}

	@Override
	public String getName() {
		return "Agglomerative";
	}
	
	@Override
	public HierarchicalClusterTree getTree() {
		return tree;
	}

	@Override
	public AgglomerativeClusterer fit() {
		synchronized(this) { // synch because alters internal structs
			
			if(null!=tree) // Then we've already fit this...
				return this;
			
			final long start = System.currentTimeMillis();
			buildTree(linkage);
			
			if(verbose) info("model " + getKey() + " completed in " + LogTimeFormatter.millis(System.currentTimeMillis()-start, false));
			return this;
			
		} // End synch
	} // End train
	
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
				tree = SingleLinkageAgglomerativeFactory
						.build(data.getData(), 
								getSeparabilityMetric(), 
								false, this);
				break;
			default:
				if(verbose) error("unimplemented linkage method");
				throw new IllegalArgumentException("unimplemented linkage method");
		}
	}

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.AGGLOMERATIVE;
	}
}
