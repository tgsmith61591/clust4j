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
import com.clust4j.utils.MatUtils;

/**
 * A builder class to provide an easier constructing
 * interface to set custom parameters for DBSCAN
 * @author Taylor G Smith
 */
final public class MeanShiftParameters 
		extends BaseClustererParameters 
		implements UnsupervisedClassifierParameters<MeanShift> {

	private static final long serialVersionUID = -2276248235151049820L;
	private boolean autoEstimateBW = false;
	private double autoEstimateBWQuantile = 0.3;
	private double bandwidth = MeanShift.DEF_BANDWIDTH;
	private int maxIter = MeanShift.DEF_MAX_ITER;
	private double minChange = MeanShift.DEF_TOL;
	private double[][] seeds = null;
	
	
	public MeanShiftParameters() {
		this.autoEstimateBW = true;
	}
	
	public MeanShiftParameters(final double bandwidth) {
		this.bandwidth = bandwidth;
	}
	

	
	public boolean getAutoEstimate() {
		return autoEstimateBW;
	}
	
	public double getAutoEstimationQuantile() {
		return autoEstimateBWQuantile;
	}
	
	public double getBandwidth() {
		return bandwidth;
	}
	
	public double[][] getSeeds() {
		return seeds;
	}
	
	public int getMaxIter() {
		return maxIter;
	}
	
	public double getConvergenceTolerance() {
		return minChange;
	}
	
	@Override
	public MeanShift fitNewModel(RealMatrix data) {
		return new MeanShift(data, this.copy()).fit();
	}
	
	@Override
	public MeanShiftParameters copy() {
		return new MeanShiftParameters(bandwidth)
			.setAutoBandwidthEstimation(autoEstimateBW)
			.setAutoBandwidthEstimationQuantile(autoEstimateBWQuantile)
			.setMaxIter(maxIter)
			.setMinChange(minChange)
			.setSeed(seed)
			.setSeeds(seeds)
			.setMetric(metric)
			.setVerbose(verbose)
			.setForceParallel(parallel);
	}
	
	public MeanShiftParameters setAutoBandwidthEstimation(boolean b) {
		this.autoEstimateBW = b;
		return this;
	}
	
	public MeanShiftParameters setAutoBandwidthEstimationQuantile(double d) {
		this.autoEstimateBWQuantile = d;
		return this;
	}
	
	public MeanShiftParameters setMaxIter(final int max) {
		this.maxIter = max;
		return this;
	}
	
	public MeanShiftParameters setMinChange(final double min) {
		this.minChange = min;
		return this;
	}
	
	@Override
	public MeanShiftParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	public MeanShiftParameters setSeeds(final double[][] seeds) {
		if(null != seeds)
			this.seeds = MatUtils.copy(seeds);
		return this;
	}
	
	@Override
	public MeanShiftParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	@Override
	public MeanShiftParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}
	
	@Override
	public MeanShiftParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
}
