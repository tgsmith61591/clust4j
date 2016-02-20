package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.SupervisedClassifierPlanner;
import com.clust4j.algo.preprocess.PreProcessor;

public class SupervisedPipeline extends Pipeline<SupervisedClassifierPlanner> {
	private static final long serialVersionUID = 8790601917700667359L;

	public SupervisedPipeline(final SupervisedClassifierPlanner planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}
	
	public AbstractClusterer fit(final AbstractRealMatrix data, int[] y) {
		AbstractRealMatrix copy = (AbstractRealMatrix)data.copy();
		
		// Push through pipeline...
		for(PreProcessor pre: pipe)
			copy = pre.operate(copy);

		// Build the model -- the model should handle the dim check internally
		final AbstractClusterer model = planner.buildNewModelInstance(copy, y);
		
		// Fit the model
		return model.fit();
	}
}