package com.clust4j.algo;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.metrics.EvaluationMetric;
import com.clust4j.utils.Classifier;

public abstract class AbstractAutonomousClusterer extends AbstractClusterer implements Classifier {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4704891508225126315L;

	public AbstractAutonomousClusterer(AbstractRealMatrix data, AbstractClusterer.BaseClustererPlanner planner) {
		super(data, planner);
	}
	
	/**
	 * The number of clusters this algorithm identified
	 * @return the number of clusters in the system
	 */
	abstract public int getNumberOfIdentifiedClusters();
	

	
	@Override
	public double score(final int[] actual) {
		return score(actual, Classifier.DEF_METRIC);
	}
	
	@Override
	public double score(final int[] actual, EvaluationMetric metric) {
		final int[] predicted = getLabels(); // Propagates a model not fit exception if not fit...
		
		if(predicted.length != actual.length)
			throw new DimensionMismatchException(actual.length, predicted.length);
		
		return metric.evaluate(actual, predicted);
	}
}
