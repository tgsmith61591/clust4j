package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

public class NearestCentroidParameters 
		extends BaseClustererParameters 
		implements SupervisedClassifierParameters<NearestCentroid> {
	
	private static final long serialVersionUID = -2064678309873097219L;
	private Double shrinkage = null;

	public NearestCentroidParameters() {
	}

	@Override
	public NearestCentroid fitNewModel(AbstractRealMatrix data, int[] y) {
		return new NearestCentroid(data, y, copy()).fit();
	}

	@Override
	public NearestCentroidParameters copy() {
		return new NearestCentroidParameters()
				.setNormalizer(norm)
				.setScale(scale)
				.setSeed(seed)
				.setMetric(metric)
				.setShrinkage(shrinkage)
				.setVerbose(verbose)
				.setForceParallel(parallel);
	}
	
	public Double getShrinkage() {
		return shrinkage;
	}

	@Override
	public NearestCentroidParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}

	@Override
	public NearestCentroidParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}

	@Override
	public NearestCentroidParameters setScale(boolean b) {
		this.scale = b;
		return this;
	}

	@Override
	public NearestCentroidParameters setSeed(Random rand) {
		this.seed = rand;
		return this;
	}

	public NearestCentroidParameters setShrinkage(final Double d) {
		this.shrinkage = d;
		return this;
	}

	@Override
	public NearestCentroidParameters setVerbose(boolean b) {
		this.verbose = b;
		return this;
	}

	@Override
	public NearestCentroidParameters setMetric(GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}

}
