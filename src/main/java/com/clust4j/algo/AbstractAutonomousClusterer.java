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

import com.clust4j.metrics.scoring.SupervisedMetric;
import static com.clust4j.metrics.scoring.UnsupervisedMetric.SILHOUETTE;

public abstract class AbstractAutonomousClusterer extends AbstractClusterer implements UnsupervisedClassifier {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4704891508225126315L;

	public AbstractAutonomousClusterer(RealMatrix data, BaseClustererParameters planner) {
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
		return SupervisedMetric.INDEX_AFFINITY.evaluate(labels, getLabels());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		// Propagates ModelNotFitException
		return SILHOUETTE.evaluate(this, getLabels());
	}
}
