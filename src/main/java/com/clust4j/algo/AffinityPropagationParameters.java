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

import org.apache.commons.math3.linear.RealMatrix;

import com.clust4j.metrics.pairwise.GeometricallySeparable;

/**
 * A model setup class for {@link AffinityPropagation}. This class houses all
 * of the hyper-parameter settings to build an {@link AffinityPropagation} instance
 * using the {@link #fitNewModel(RealMatrix)} method.
 * @author Taylor G Smith
 */
public class AffinityPropagationParameters 
			extends BaseClustererParameters 
			implements UnsupervisedClassifierParameters<AffinityPropagation> {
	
	private static final long serialVersionUID = -6096855634412545959L;
	protected int maxIter = AffinityPropagation.DEF_MAX_ITER;
	protected double minChange = AffinityPropagation.DEF_TOL;
	protected int iterBreak = AffinityPropagation.DEF_ITER_BREAK;
	protected double damping = AffinityPropagation.DEF_DAMPING;
	protected boolean addNoise = AffinityPropagation.DEF_ADD_GAUSSIAN_NOISE;

	public AffinityPropagationParameters() { /* Default constructor */ }
	public AffinityPropagationParameters useGaussianSmoothing(boolean b) {
		this.addNoise = b;
		return this;
	}

	@Override
	public AffinityPropagation fitNewModel(RealMatrix data) {
		return new AffinityPropagation(data, this.copy()).fit();
	}
	
	@Override
	public AffinityPropagationParameters copy() {
		return new AffinityPropagationParameters()
			.setDampingFactor(damping)
			.setIterBreak(iterBreak)
			.setMaxIter(maxIter)
			.setMinChange(minChange)
			.setSeed(seed)
			.setMetric(metric)
			.setVerbose(verbose)
			.useGaussianSmoothing(addNoise)
			.setForceParallel(parallel);
	}
	
	public AffinityPropagationParameters setDampingFactor(final double damp) {
		this.damping = damp;
		return this;
	}
	
	public AffinityPropagationParameters setIterBreak(final int iters) {
		this.iterBreak = iters;
		return this;
	}
	
	public AffinityPropagationParameters setMaxIter(final int max) {
		this.maxIter = max;
		return this;
	}
	
	public AffinityPropagationParameters setMinChange(final double min) {
		this.minChange = min;
		return this;
	}

	@Override
	public AffinityPropagationParameters setSeed(Random rand) {
		seed = rand;
		return this;
	}
	
	@Override
	public AffinityPropagationParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}

	@Override
	public AffinityPropagationParameters setVerbose(boolean b) {
		verbose = b;
		return this;
	}

	@Override
	public AffinityPropagationParameters setMetric(GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
}