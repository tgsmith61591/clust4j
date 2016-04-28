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
import com.clust4j.metrics.scoring.UnsupervisedMetric;

import static com.clust4j.metrics.scoring.UnsupervisedMetric.SILHOUETTE;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface BaseClassifier extends java.io.Serializable {
	public final static SupervisedMetric DEF_SUPERVISED_METRIC = SupervisedMetric.BINOMIAL_ACCURACY;
	public final static UnsupervisedMetric DEF_UNSUPERVISED_METRIC = SILHOUETTE;
	
	/**
	 * Returns a copy of the assigned class labels in
	 * record order
	 * @return
	 */
	public int[] getLabels();
	
	/**
	 * Predict on new data
	 * @param newData
	 * @throws ModelNotFitException if the model hasn't yet been fit
	 * @return
	 */
	public int[] predict(RealMatrix newData);
}
