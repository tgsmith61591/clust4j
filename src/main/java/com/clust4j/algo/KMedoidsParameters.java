/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/

package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

public class KMedoidsParameters extends CentroidClustererParameters<KMedoids> {
	private static final long serialVersionUID = -3288579217568576647L;
	
	private InitializationStrategy strat = KMedoids.DEF_INIT;
	private int maxIter = KMedoids.DEF_MAX_ITER;
	
	public KMedoidsParameters() { }
	public KMedoidsParameters(int k) {
		this.k = k;
	}
	
	@Override
	public KMedoids fitNewModel(final AbstractRealMatrix data) {
		return new KMedoids(data, this.copy()).fit();
	}
	
	@Override
	public KMedoidsParameters copy() {
		return new KMedoidsParameters(k)
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
	public KMedoidsParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
	
	@Override
	public KMedoidsParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	public KMedoidsParameters setMaxIter(final int max) {
		this.maxIter = max;
		return this;
	}

	@Override
	public KMedoidsParameters setConvergenceCriteria(final double min) {
		this.minChange = min;
		return this;
	}
	
	@Override
	public KMedoidsParameters setInitializationStrategy(InitializationStrategy init) {
		this.strat = init;
		return this;
	}
	
	@Override
	public KMedoidsParameters setScale(final boolean scale) {
		this.scale = scale;
		return this;
	}
	
	@Override
	public KMedoidsParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	@Override
	public KMedoidsParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}

	@Override
	public KMedoidsParameters setNormalizer(FeatureNormalization norm) {
		this.norm = norm;
		return this;
	}
}