package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractDBSCAN.AbstractDBSCANPlanner;
import com.clust4j.algo.HDBSCAN.HDBSCAN_Algorithm;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

/**
 * A builder class to provide an easier constructing
 * interface to set custom parameters for HDBSCAN
 * @author Taylor G Smith
 */
final public class HDBSCANParameters extends AbstractDBSCANPlanner<HDBSCAN> {
	private static final long serialVersionUID = 7197585563308908685L;
	
	private int minPts = HDBSCAN.DEF_MIN_PTS;
	private HDBSCAN_Algorithm algo = HDBSCAN.DEF_ALGO;
	private double alpha = HDBSCAN.DEF_ALPHA;
	private boolean approxMinSpanTree = HDBSCAN.DEF_APPROX_MIN_SPAN;
	private int min_cluster_size = HDBSCAN.DEF_MIN_CLUST_SIZE;
	private int leafSize = HDBSCAN.DEF_LEAF_SIZE;
	
	
	public HDBSCANParameters() { this(HDBSCAN.DEF_MIN_PTS); }
	public HDBSCANParameters(final int minPts) {
		this.minPts = minPts;
	}

	
	@Override
	public HDBSCAN fitNewModel(AbstractRealMatrix data) {
		return new HDBSCAN(data, this.copy()).fit();
	}
	
	@Override
	public HDBSCANParameters copy() {
		return new HDBSCANParameters(minPts)
			.setAlgo(algo)
			.setAlpha(alpha)
			.setApprox(approxMinSpanTree)
			.setLeafSize(leafSize)
			.setMinClustSize(min_cluster_size)
			.setMinPts(minPts)
			.setScale(scale)
			.setMetric(metric)
			.setSeed(seed)
			.setVerbose(verbose)
			.setNormalizer(norm)
			.setForceParallel(parallel);
	}

	@Override
	public int getMinPts() {
		return minPts;
	}
	
	public HDBSCAN_Algorithm getAlgo() {
		return this.algo;
	}
	
	public HDBSCANParameters setAlgo(final HDBSCAN_Algorithm algo) {
		this.algo = algo;
		return this;
	}
	
	public double getAlpha() {
		return alpha;
	}
	
	public HDBSCANParameters setAlpha(final double a) {
		this.alpha = a;
		return this;
	}
	
	public boolean getApprox() {
		return approxMinSpanTree;
	}
	
	public HDBSCANParameters setApprox(final boolean b) {
		this.approxMinSpanTree = b;
		return this;
	}
	
	public int getLeafSize() {
		return leafSize;
	}
	
	public HDBSCANParameters setLeafSize(final int leafSize) {
		this.leafSize = leafSize;
		return this;
	}
	
	public int getMinClusterSize() {
		return min_cluster_size;
	}
	
	public HDBSCANParameters setMinClustSize(final int min) {
		this.min_cluster_size = min;
		return this;
	}
	
	@Override
	public HDBSCANParameters setMinPts(final int minPts) {
		this.minPts = minPts;
		return this;
	}
	
	@Override
	public HDBSCANParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
	
	@Override
	public HDBSCANParameters setScale(final boolean scale) {
		this.scale = scale;
		return this;
	}
	
	@Override
	public HDBSCANParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	@Override
	public HDBSCANParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	public HDBSCANParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}
	
	@Override
	public HDBSCANParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}
}
