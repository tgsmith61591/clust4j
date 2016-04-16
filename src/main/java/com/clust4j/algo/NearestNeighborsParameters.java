package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.BaseNeighborsModel.BaseNeighborsPlanner;
import com.clust4j.algo.BaseNeighborsModel.NeighborsAlgorithm;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

public class NearestNeighborsParameters extends BaseNeighborsPlanner<NearestNeighbors> {
	private static final long serialVersionUID = -4848896423352149405L;
	private final int k;
	
	
	public NearestNeighborsParameters() { this(BaseNeighborsModel.DEF_K); }
	public NearestNeighborsParameters(int k) {
		this.k = k;
	}
	
	@Override
	public NearestNeighbors fitNewModel(AbstractRealMatrix data) {
		return new NearestNeighbors(data, this.copy()).fit();
	}

	@Override
	public NearestNeighborsParameters setAlgorithm(NeighborsAlgorithm algo) {
		this.algo = algo;
		return this;
	}

	@Override
	public NearestNeighborsParameters copy() {
		return new NearestNeighborsParameters(k)
			.setAlgorithm(algo)
			.setNormalizer(norm)
			.setScale(scale)
			.setSeed(seed)
			.setMetric(metric)
			.setVerbose(verbose)
			.setLeafSize(leafSize)
			.setForceParallel(parallel);
	}
	
	@Override
	final public Integer getK() {
		return k;
	}

	@Override
	final public Double getRadius() {
		return null;
	}

	public NearestNeighborsParameters setLeafSize(int leafSize) {
		this.leafSize = leafSize;
		return this;
	}
	
	@Override
	public NearestNeighborsParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}

	@Override
	public NearestNeighborsParameters setScale(boolean b) {
		this.scale = b;
		return this;
	}

	@Override
	public NearestNeighborsParameters setSeed(Random rand) {
		this.seed= rand;
		return this;
	}

	@Override
	public NearestNeighborsParameters setVerbose(boolean b) {
		this.verbose = b;
		return this;
	}

	@Override
	public NearestNeighborsParameters setMetric(GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	@Override
	public NearestNeighborsParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
}
