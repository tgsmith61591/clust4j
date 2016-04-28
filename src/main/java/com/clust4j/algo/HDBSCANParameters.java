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

import com.clust4j.algo.AbstractDBSCAN.AbstractDBSCANParameters;
import com.clust4j.algo.HDBSCAN.HDBSCAN_Algorithm;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

/**
 * A builder class to provide an easier constructing
 * interface to set custom parameters for HDBSCAN
 * @author Taylor G Smith
 */
final public class HDBSCANParameters extends AbstractDBSCANParameters<HDBSCAN> {
	private static final long serialVersionUID = 7197585563308908685L;
	
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
	public HDBSCAN fitNewModel(RealMatrix data) {
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
			.setMetric(metric)
			.setSeed(seed)
			.setVerbose(verbose)
			.setForceParallel(parallel);
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
}
