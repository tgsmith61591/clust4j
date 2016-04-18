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

import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.metrics.scoring.UnsupervisedMetric;

public interface UnsupervisedClassifier extends BaseClassifier {
	/**
	 * Evaluate how the model performed against a truth set. This method
	 * utilizes the {@link SupervisedMetric#INDEX_AFFINITY} class
	 * @param actualLabels
	 * @return
	 */
	public double indexAffinityScore(int[] labels);
	
	
	/**
	 * Evaluate how the model performed via the {@link UnsupervisedMetric#SILHOUETTE} metric
	 * @return
	 */
	public double silhouetteScore();
}
