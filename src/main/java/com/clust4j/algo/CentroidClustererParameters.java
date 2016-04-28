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

import org.apache.commons.math3.linear.RealMatrix;

import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;

public abstract class CentroidClustererParameters<T extends AbstractCentroidClusterer> extends BaseClustererParameters
		implements UnsupervisedClassifierParameters<T>, ConvergeablePlanner {

	private static final long serialVersionUID = -1984508955251863189L;
	protected int k = AbstractCentroidClusterer.DEF_K;
	protected double minChange = AbstractCentroidClusterer.DEF_CONVERGENCE_TOLERANCE;

	@Override abstract public T fitNewModel(RealMatrix mat);
	@Override abstract public int getMaxIter();
	abstract public InitializationStrategy getInitializationStrategy();
	abstract public CentroidClustererParameters<T> setConvergenceCriteria(final double min);
	abstract public CentroidClustererParameters<T> setInitializationStrategy(final InitializationStrategy strat);

	final public int getK() {
		return k;
	}

	@Override
	final public double getConvergenceTolerance() {
		return minChange;
	}
}
