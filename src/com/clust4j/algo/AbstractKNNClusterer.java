package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.Classifier;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.SupervisedLearner;

public abstract class AbstractKNNClusterer extends AbstractPartitionalClusterer implements SupervisedLearner, Classifier {

	public AbstractKNNClusterer(AbstractRealMatrix data, KNNPlanner planner) {
		super(data, planner, planner.k);
	}

	public static class KNNPlanner extends AbstractClusterer.BaseClustererPlanner {
		protected GeometricallySeparable dist = DEF_DIST;
		protected boolean verbose = DEF_VERBOSE;
		protected boolean scale = DEF_SCALE;
		protected int k;
		
		public KNNPlanner(final int k) {
			this.k = k;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public KNNPlanner setDist(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		@Override
		public KNNPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}

		@Override
		public GeometricallySeparable getDist() {
			return dist;
		}

		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public KNNPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
	}
}
