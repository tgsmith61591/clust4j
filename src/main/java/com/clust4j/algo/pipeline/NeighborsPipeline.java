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

import com.clust4j.algo.BaseNeighborsModel;
import com.clust4j.algo.NeighborsClassifierParameters;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.except.ModelNotFitException;

public class NeighborsPipeline<M extends BaseNeighborsModel> 
		extends Pipeline<NeighborsClassifierParameters<M>> {

	private static final long serialVersionUID = 7363030699567515649L;
	protected M fit_model = null;

	public NeighborsPipeline(final NeighborsClassifierParameters<M> planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}
	
	public M fit(final RealMatrix data) {
		synchronized(fitLock) {
			RealMatrix copy = pipelineFitTransform(data);
	
			// Build/fit the model -- the model should handle the dim check internally
			return fit_model = planner.fitNewModel(copy);
		}
	}
	
	@Override
	protected void checkFit() {
		if(null == fit_model)
			throw new ModelNotFitException("model not yet fit");
	}
}
