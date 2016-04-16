package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

final public class KMeansParameters extends CentroidClustererParameters<KMeans> {
	private static final long serialVersionUID = -813106538623499760L;
	
	private InitializationStrategy strat = KMeans.DEF_INIT;
	private int maxIter = KMeans.DEF_MAX_ITER;
	
	public KMeansParameters() { }
	public KMeansParameters(int k) {
		this.k = k;
	}
	
	@Override
	public KMeans fitNewModel(final AbstractRealMatrix data) {
		return new KMeans(data, this.copy()).fit();
	}
	
	@Override
	public KMeansParameters copy() {
		return new KMeansParameters(k)
			.setMaxIter(maxIter)
			.setConvergenceCriteria(minChange)
			.setScale(scale)
			.setMetric(metric)
			.setVerbose(verbose)
			.setSeed(seed)
			.setNormalizer(norm)
			.setInitializationStrategy(strat)
			.setForceParallel(parallel);
	}
	
	@Override
	public InitializationStrategy getInitializationStrategy() {
		return strat;
	}
	
	@Override
	public int getMaxIter() {
		return maxIter;
	}
	
	@Override
	public KMeansParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
	
	@Override
	public KMeansParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	public KMeansParameters setMaxIter(final int max) {
		this.maxIter = max;
		return this;
	}

	@Override
	public KMeansParameters setConvergenceCriteria(final double min) {
		this.minChange = min;
		return this;
	}
	
	@Override
	public KMeansParameters setInitializationStrategy(InitializationStrategy init) {
		this.strat = init;
		return this;
	}
	
	@Override
	public KMeansParameters setScale(final boolean scale) {
		this.scale = scale;
		return this;
	}
	
	@Override
	public KMeansParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	@Override
	public KMeansParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}

	@Override
	public KMeansParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}
}
