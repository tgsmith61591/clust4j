package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.GeometricallySeparable;

public abstract class AbstractHierarchicalClusterer extends AbstractClusterer {
	public AbstractHierarchicalClusterer(AbstractRealMatrix data, BaseHierarchicalPlanner planner) {
		super(data, planner);
	}
	
	public static class BaseHierarchicalPlanner extends AbstractClusterer.BaseClustererPlanner {
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;

		@Override
		public GeometricallySeparable getDist() {
			return dist;
		}

		@Override
		public boolean getScale() {
			return scale;
		}

		@Override
		public BaseClustererPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}

		@Override
		public BaseClustererPlanner setDist(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
	}
}
