package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.AbstractClusterer.BaseClustererPlanner;
import com.clust4j.algo.prep.PreProcessor;

public class Pipeline {
	final private BaseClustererPlanner planner;
	final private PreProcessor[] pipe;
	
	public Pipeline(final BaseClustererPlanner planner, final PreProcessor... pipe) {
		this.planner = planner.copy();
		this.pipe = copyPipe(pipe);
	}
	
	final static PreProcessor[] copyPipe(final PreProcessor... pipe) {
		final PreProcessor[] out = new PreProcessor[pipe.length];
		
		int idx = 0;
		for(PreProcessor pre: pipe)
			out[idx++] = pre.copy();
		
		return out;
	}
	
	public AbstractClusterer fit(final AbstractRealMatrix data) {
		AbstractRealMatrix copy = (AbstractRealMatrix)data.copy();
		
		// Push through pipeline...
		for(PreProcessor pre: pipe)
			copy = pre.operate(copy);

		// Build the model
		final AbstractClusterer model = planner.buildNewModelInstance(copy);
		
		// Fit the model
		return model.fit();
	}
}
