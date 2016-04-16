package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;

public abstract class CentroidClustererParameters<T extends AbstractCentroidClusterer> extends BaseClustererParameters
		implements UnsupervisedClassifierParameters<T>, ConvergeablePlanner {

	private static final long serialVersionUID = -1984508955251863189L;
	protected int k = AbstractCentroidClusterer.DEF_K;
	protected double minChange = AbstractCentroidClusterer.DEF_CONVERGENCE_TOLERANCE;

	@Override abstract public T fitNewModel(AbstractRealMatrix mat);
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
