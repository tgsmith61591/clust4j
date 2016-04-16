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

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.UnsupervisedClassifier;
import com.clust4j.algo.UnsupervisedClassifierParameters;
import com.clust4j.algo.preprocess.PreProcessor;

public class UnsupervisedPipeline<M extends AbstractClusterer & UnsupervisedClassifier> 
		extends Pipeline<UnsupervisedClassifierParameters<M>> {
	private static final long serialVersionUID = 8790601917700667359L;

	public UnsupervisedPipeline(final UnsupervisedClassifierParameters<M> planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}

	public M fit(final AbstractRealMatrix data) {
		synchronized(fitLock) {
			AbstractRealMatrix copy = data;
			
			// Push through pipeline...
			for(PreProcessor pre: pipe)
				copy = pre.operate(copy);
	
			// Build the model
			final M model = planner.fitNewModel(copy);
			
			// The model was fit internally above
			return model;
		}
	}
}
