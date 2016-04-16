package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractDBSCAN.AbstractDBSCANPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

/**
 * A builder class to provide an easier constructing
 * interface to set custom parameters for DBSCAN
 * @author Taylor G Smith
 */
final public class DBSCANParameters extends AbstractDBSCANPlanner<DBSCAN> {
	private static final long serialVersionUID = -5285244186285768512L;
	
	private double eps = DBSCAN.DEF_EPS;
	private int minPts = DBSCAN.DEF_MIN_PTS;
	
	
	public DBSCANParameters() { }
	public DBSCANParameters(final double eps) {
		this.eps = eps;
	}

	
	@Override
	public DBSCAN fitNewModel(AbstractRealMatrix data) {
		return new DBSCAN(data, this.copy()).fit();
	}
	
	@Override
	public DBSCANParameters copy() {
		return new DBSCANParameters(eps)
			.setMinPts(minPts)
			.setScale(scale)
			.setMetric(metric)
			.setSeed(seed)
			.setVerbose(verbose)
			.setNormalizer(norm)
			.setForceParallel(parallel);
	}
	
	public double getEps() {
		return eps;
	}

	@Override
	public int getMinPts() {
		return minPts;
	}
	
	public DBSCANParameters setEps(final double eps) {
		this.eps = eps;
		return this;
	}
	
	@Override
	public DBSCANParameters setMinPts(final int minPts) {
		this.minPts = minPts;
		return this;
	}
	
	@Override
	public DBSCANParameters setScale(final boolean scale) {
		this.scale = scale;
		return this;
	}
	
	@Override
	public DBSCANParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	@Override
	public DBSCANParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	public DBSCANParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}
	
	@Override
	public DBSCANParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}
	
	@Override
	public DBSCANParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
}