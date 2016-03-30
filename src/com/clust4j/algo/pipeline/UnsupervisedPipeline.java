package com.clust4j.algo.pipeline;

import lombok.Synchronized;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.UnsupervisedClassifierPlanner;
import com.clust4j.algo.preprocess.PreProcessor;

public class UnsupervisedPipeline extends Pipeline<UnsupervisedClassifierPlanner> {
	private static final long serialVersionUID = 8790601917700667359L;

	public UnsupervisedPipeline(final UnsupervisedClassifierPlanner planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}

	@Synchronized("fitLock") 
	public AbstractClusterer fit(final AbstractRealMatrix data) {
		AbstractRealMatrix copy = data;
		
		// Push through pipeline...
		for(PreProcessor pre: pipe)
			copy = pre.operate(copy);

		// Build the model
		final AbstractClusterer model = planner.buildNewModelInstance(copy);
		
		// Fit the model
		return model.fit();
	}
}
