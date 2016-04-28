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
import com.clust4j.metrics.pairwise.GeometricallySeparable;

/**
 * A builder class to provide an easier constructing
 * interface to set custom parameters for DBSCAN
 * @author Taylor G Smith
 */
final public class DBSCANParameters extends AbstractDBSCANParameters<DBSCAN> {
	private static final long serialVersionUID = -5285244186285768512L;
	
	private double eps = DBSCAN.DEF_EPS;
	
	
	public DBSCANParameters() { }
	public DBSCANParameters(final double eps) {
		this.eps = eps;
	}

	
	@Override
	public DBSCAN fitNewModel(RealMatrix data) {
		return new DBSCAN(data, this.copy()).fit();
	}
	
	@Override
	public DBSCANParameters copy() {
		return new DBSCANParameters(eps)
			.setMinPts(minPts)
			.setMetric(metric)
			.setSeed(seed)
			.setVerbose(verbose)
			.setForceParallel(parallel);
	}
	
	public double getEps() {
		return eps;
	}
	
	public DBSCANParameters setEps(final double eps) {
		this.eps = eps;
		return this;
	}
	
	@Override
	public DBSCANParameters setMinPts(final int minPts) {
		this.minPts = minPts;
		return this;
	}
	
	@Override
	public DBSCANParameters setSeed(final Random seed) {
		this.seed = seed;
		return this;
	}
	
	@Override
	public DBSCANParameters setMetric(final GeometricallySeparable dist) {
		this.metric = dist;
		return this;
	}
	
	public DBSCANParameters setVerbose(final boolean v) {
		this.verbose = v;
		return this;
	}
	
	@Override
	public DBSCANParameters setForceParallel(boolean b) {
		this.parallel = b;
		return this;
	}
}