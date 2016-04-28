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

import com.clust4j.Clust4j;
import com.clust4j.NamedEntity;
import com.clust4j.algo.BaseClassifierParameters;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.utils.SynchronicityLock;

public abstract class Pipeline<T extends BaseClassifierParameters> 
		extends Clust4j implements NamedEntity {
	
	private static final long serialVersionUID = 3491192139356583621L;
	final Object fitLock = new SynchronicityLock();
	final PreProcessor[] pipe;
	final T planner;

	@SuppressWarnings("unchecked")
	public Pipeline(T planner, PreProcessor... pipe) {
		this.planner = (T)planner.copy();
		this.pipe = copyPipe(pipe);
	}
	
	/**
	 * Ensure the pipeline remains immutable
	 * @param pipe
	 * @return
	 */
	protected final static PreProcessor[] copyPipe(final PreProcessor... pipe) {
		final PreProcessor[] out = new PreProcessor[pipe.length];
		
		int idx = 0;
		for(PreProcessor pre: pipe)
			out[idx++] = pre.copy();
		
		return out;
	}
	
	/**
	 * Apply the pipeline to input data
	 * @param data
	 * @return
	 */
	protected final RealMatrix pipelineFitTransform(RealMatrix data) {
		RealMatrix operated = data;
		
		// Push through pipeline... fits the models in place
		for(PreProcessor pre: pipe)
			operated = pre.fit(operated).transform(operated);
		
		return operated;
	}
	
	/**
	 * Apply the pipeline to test data
	 * @param data
	 * @return
	 */
	protected final RealMatrix pipelineTransform(RealMatrix data) {
		RealMatrix operated = data;
		
		// Push through pipeline... the models are already fit...
		for(PreProcessor pre: pipe)
			operated = pre.transform(operated);
		
		return operated;
	}
	
	
	@Override
	public String getName() {
		return "Pipeline";
	}
	
	abstract protected void checkFit();
}
