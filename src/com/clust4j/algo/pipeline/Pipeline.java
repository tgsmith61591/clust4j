package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.prep.PreProcessor;

public class Pipeline<T extends AbstractClusterer> {
	final private T.BaseClustererPlanner planner;
	final private PreProcessor[] pipe;
	
	public Pipeline(final T.BaseClustererPlanner planner, final PreProcessor... pipe) {
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
	
	@SuppressWarnings("unchecked")
	public T fit(final AbstractRealMatrix data) {
		AbstractRealMatrix copy = (AbstractRealMatrix)data.copy();
		
		// Push through pipeline...
		for(PreProcessor pre: pipe)
			copy = pre.operate(copy);

		// Build the model
		final T model;
		try {
			model = (T) planner.buildNewModelInstance(copy);
		} catch(ClassCastException t) {
			throw new IllegalArgumentException("generic type T's "
				+ "planner class does not match provided planner");
		}
		
		// Fit the model
		return (T) model.fit();
	}
}
