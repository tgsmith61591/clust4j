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

import com.clust4j.metrics.pairwise.SimilarityMetric;

public abstract class AbstractDensityClusterer extends AbstractAutonomousClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5645721633522621894L;

	public AbstractDensityClusterer(RealMatrix data, BaseClustererParameters planner) {
		super(data, planner);
		
		checkState(this);
	} // End constructor
	
	protected static void checkState(AbstractClusterer ac) {
		// Should not use similarity metrics in DBClusterers, DB looks for 
		// neighborhoods not accurately represented via similarity metrics.
		if(ac.getSeparabilityMetric() instanceof SimilarityMetric) {
			ac.warn("density or radius-based clustering algorithms "
				+ "should use distance metrics instead of similarity metrics. "
				+ "Falling back to default: " + DEF_DIST);
			ac.setSeparabilityMetric(DEF_DIST);
		}
	}
}
