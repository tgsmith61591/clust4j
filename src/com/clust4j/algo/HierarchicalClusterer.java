package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.Classifier;
import com.clust4j.utils.Distance;
import com.clust4j.utils.Hierarchical;
import com.clust4j.utils.IllegalClusterStateException;

abstract public class HierarchicalClusterer extends AbstractPartitionalClusterer implements Classifier, Hierarchical {
	private static final long serialVersionUID = -1248722938839039425L;

	public static final Linkage DEF_LINKAGE = Linkage.WARD;
	
	/**
	 * Which {@link Linkage} to use for the clustering algorithm
	 */
	final protected Linkage linkage;
	
	/**
	 * The linkages for agglomerative clustering. 
	 * @author Taylor G Smith
	 *
	 */
	public enum Linkage implements java.io.Serializable {
		AVERAGE, COMPLETE, WARD
	}
	

	
	public HierarchicalClusterer(AbstractRealMatrix data, BaseHierarchicalPlanner planner, int k) {
		super(data, planner, k);
		this.linkage = planner.getLinkage();

		checkLinkage(this, linkage);
	}
	
	protected static void checkLinkage(AbstractClusterer algo, Linkage link) {
		Hierarchical hier = (Hierarchical)algo;
		Linkage linkage = hier.getLinkage();
		if(null == linkage) {
			String e = "null linkage passed to planner";
			algo.error(e);
			throw new IllegalClusterStateException(e);
		} else if(linkage.equals(Linkage.WARD) && !algo.getSeparabilityMetric().equals(Distance.EUCLIDEAN)) {
			algo.warn("Ward's method implicitly requires Euclidean distance; overriding " + 
					algo.getSeparabilityMetric().getName());
			
			algo.setSeparabilityMetric(Distance.EUCLIDEAN);
			algo.meta("New distance metric: "+algo.getSeparabilityMetric().getName());
		}
	}
	
	abstract public static class BaseHierarchicalPlanner extends BaseClustererPlanner implements Hierarchical {
		@Override abstract public Linkage getLinkage();
		abstract public BaseHierarchicalPlanner setLinkage(Linkage linkage);
	}
	
	@Override
	public Linkage getLinkage() {
		return linkage;
	}
}
