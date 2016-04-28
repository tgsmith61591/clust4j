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

import com.clust4j.algo.BaseNeighborsModel.BaseNeighborsPlanner;
import com.clust4j.algo.BaseNeighborsModel.NeighborsAlgorithm;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

public class RadiusNeighborsParameters extends BaseNeighborsPlanner<RadiusNeighbors> {
	private static final long serialVersionUID = 2183556008789826257L;
	private double radius;
	
	public RadiusNeighborsParameters() { this(RadiusNeighbors.DEF_RADIUS); }
	public RadiusNeighborsParameters(double rad) {
		this.radius = rad;
	}

	
	@Override
	public RadiusNeighbors fitNewModel(RealMatrix data) {
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
