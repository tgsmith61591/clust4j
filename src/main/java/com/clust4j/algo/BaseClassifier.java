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

import com.clust4j.metrics.scoring.BinomialClassificationScoring;
import com.clust4j.metrics.scoring.SilhouetteScore;
import com.clust4j.metrics.scoring.SupervisedEvaluationMetric;
import com.clust4j.metrics.scoring.UnsupervisedEvaluationMetric;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface BaseClassifier extends java.io.Serializable {
	public final static SupervisedEvaluationMetric DEF_SUPERVISED_METRIC = BinomialClassificationScoring.ACCURACY;
	public final static UnsupervisedEvaluationMetric DEF_UNSUPERVISED_METRIC = SilhouetteScore.getInstance();
	
	/**
	 * Returns a copy of the assigned class labels in
	 * record order
	 * @return
	 */
	public int[] getLabels();
}
