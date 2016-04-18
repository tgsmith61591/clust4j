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

public interface SupervisedClassifier extends BaseClassifier {
	public int[] getTrainingLabels();
	
	
	/**
	 * Evaluate how the model performed. Every classifier should
	 * have a default scoring method
	 * @param actualLabels
	 * @return
	 */
	public double score();
	
	/**
	 * Evaluate how the model performed
	 * @param actualLabels
	 * @return
	 */
	public double score(final SupervisedMetric metric);
}
