package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.BaseNeighborsModel.BaseNeighborsPlanner;
import com.clust4j.algo.BaseNeighborsModel.NeighborsAlgorithm;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

public class RadiusNeighborsParameters extends BaseNeighborsPlanner<RadiusNeighbors> {
	private static final long serialVersionUID = 2183556008789826257L;
	private double radius;
	
	public RadiusNeighborsParameters() { this(RadiusNeighbors.DEF_RADIUS); }
	public RadiusNeighborsParameters(double rad) {
		this.radius = rad;
	}

	
	@Override
	public RadiusNeighbors fitNewModel(AbstractRealMatrix data) {
		return new RadiusNeighbors(data, this.copy()).fit();
	}

	@Override
	public RadiusNeighborsParameters setAlgorithm(NeighborsAlgorithm algo) {
		this.algo = algo;
		return this;
	}

	@Override
	public RadiusNeighborsParameters copy() {
		return new RadiusNeighborsParameters(radius)
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
		return null;
	}

	@Override
	final public Double getRadius() {
		return radius;
	}

	public RadiusNeighborsParameters setLeafSize(int leafSize) {
		this.leafSize = leafSize;
		return this;
	}
	
	@Override
	public RadiusNeighborsParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}

	@Override
	public RadiusNeighborsParameters setScale(boolean b) {
		this.scale = b;
		return this;
	}

	@Override
	public RadiusNeighborsParameters setSeed(Random rand) {
		this.seed= rand;
		return this;
	}

	@Override
	public RadiusNeighborsParameters setVerbose(boolean b) {
		this.verbose = b;
		return this;
	}

	@Override
	public RadiusNeighborsParameters setMetric(GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	@Override
	public RadiusNeighborsParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
}
