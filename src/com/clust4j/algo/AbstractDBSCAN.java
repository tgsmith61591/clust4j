package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.utils.NoiseyClusterer;

abstract class AbstractDBSCAN extends AbstractDensityClusterer implements NoiseyClusterer {
	private static final long serialVersionUID = 5247910788105653778L;
	
	final public static double DEF_EPS = 0.5;
	final public static int DEF_MIN_PTS = 5;
	final public static int NOISE_CLASS = -1;

	final protected int minPts;
	protected double eps = DEF_EPS;
	final protected FeatureNormalization normer;

	public AbstractDBSCAN(AbstractRealMatrix data, AbstractDBSCANPlanner planner) {
		super(data, planner);
		
		this.minPts = planner.getMinPts();
		this.normer = planner.getNormalizer();
		
		String e;
		if(this.minPts < 1) {
			e="minPts must be greater than 0";
			error(e);
			throw new IllegalArgumentException(e);
		}
	}
	
	abstract public static class AbstractDBSCANPlanner extends AbstractClusterer.BaseClustererPlanner {
		abstract public AbstractDBSCANPlanner setMinPts(final int minPts);
		abstract public int getMinPts();
	}
	
	public int getMinPts() {
		return minPts;
	}
}
