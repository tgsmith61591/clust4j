package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.scoring.SilhouetteScore;
import com.clust4j.metrics.scoring.UnsupervisedIndexAffinity;

public abstract class AbstractAutonomousClusterer extends AbstractClusterer implements UnsupervisedClassifier {
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
	
	
	/** {@inheritDoc} */
	@Override
	public double indexAffinityScore(int[] labels) {
		// Propagates ModelNotFitException
		return UnsupervisedIndexAffinity.getInstance().evaluate(labels, getLabels());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		return silhouetteScore(getSeparabilityMetric());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore(GeometricallySeparable dist) {
		// Propagates ModelNotFitException
		return SilhouetteScore.getInstance().evaluate(this, dist, getLabels());
	}
}
