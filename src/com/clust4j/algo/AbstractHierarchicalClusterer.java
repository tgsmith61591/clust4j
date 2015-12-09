package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.HierarchicalClusterTree;
import com.clust4j.utils.Linkage;

public abstract class AbstractHierarchicalClusterer extends AbstractClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2485741842587360171L;
	public static final Linkage DEF_LINKAGE = Linkage.SINGLE;
	protected final Linkage linkage;
	
	public AbstractHierarchicalClusterer(AbstractRealMatrix data, BaseHierarchicalPlanner planner) {
		super(data, planner);
		this.linkage = planner.linkage;
	}
	
	public static class BaseHierarchicalPlanner extends AbstractClusterer.BaseClustererPlanner {
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private Linkage linkage = DEF_LINKAGE;
		private boolean verbose = DEF_VERBOSE;

		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}

		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		public BaseHierarchicalPlanner setLinkage(Linkage l) {
			this.linkage = l;
			return this;
		}

		@Override
		public BaseHierarchicalPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}
		
		@Override
		public BaseHierarchicalPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public BaseHierarchicalPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}

		@Override
		public BaseHierarchicalPlanner setSep(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
	}

	
	@Override
	public String toString() {
		if(null == getTree()) return super.toString();
		return super.toString() + ": " + getTree().toString();
	}
	
	/**
	 * Get the tree generated from splits or merges
	 * @return the hierarchical tree
	 */
	public abstract HierarchicalClusterTree getTree();
}
