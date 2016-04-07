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

import lombok.Synchronized;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.SupervisedClassifierPlanner;
import com.clust4j.algo.preprocess.PreProcessor;

public class SupervisedPipeline extends Pipeline<SupervisedClassifierPlanner> {
	private static final long serialVersionUID = 8790601917700667359L;

	public SupervisedPipeline(final SupervisedClassifierPlanner planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}

	@Synchronized("fitLock") 
	public AbstractClusterer fit(final AbstractRealMatrix data, int[] y) {
		AbstractRealMatrix copy = data; // no need to copy, as handled in each processor...
		
		// Push through pipeline...
		for(PreProcessor pre: pipe)
			copy = pre.operate(copy);

		// Build the model -- the model should handle the dim check internally
		final AbstractClusterer model = planner.buildNewModelInstance(copy, y);
		
		// Fit the model
		return model.fit();
	}
}