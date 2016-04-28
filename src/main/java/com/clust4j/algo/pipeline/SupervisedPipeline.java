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
package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.RealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.SupervisedClassifier;
import com.clust4j.algo.SupervisedClassifierParameters;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.metrics.scoring.SupervisedMetric;

public class SupervisedPipeline<M extends AbstractClusterer & SupervisedClassifier> 
		extends Pipeline<SupervisedClassifierParameters<M>>
		implements SupervisedClassifier {
	
	private static final long serialVersionUID = 8790601917700667359L;
	protected M fit_model = null;

	public SupervisedPipeline(final SupervisedClassifierParameters<M> planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}

	public M fit(final RealMatrix data, int[] y) {
		synchronized(fitLock) {
			RealMatrix copy = pipelineFitTransform(data);
	
			// Build/fit the model -- the model should handle the dim check internally
			return fit_model = planner.fitNewModel(copy, y);
		}
	}

	@Override
	public int[] getLabels() {
		checkFit();
		return fit_model.getLabels();
	}

	@Override
	public int[] getTrainingLabels() {
		checkFit();
		return fit_model.getTrainingLabels();
	}

	@Override
	public double score() {
		checkFit();
		return fit_model.score();
	}

	@Override
	public double score(SupervisedMetric metric) {
		checkFit();
		return fit_model.score(metric);
	}

	/**
	 * Given an incoming dataframe, pipeline transform and
	 * predict via the fit model
	 * @param newData
	 */
	@Override
	public int[] predict(RealMatrix newData) {
		checkFit();
		return fit_model.predict(pipelineTransform(newData));
	}
	
	@Override
	protected void checkFit() {
		if(null == fit_model)
			throw new ModelNotFitException("model not yet fit");
	}
}