package com.clust4j.algo.pipeline;

import com.clust4j.Clust4j;
import com.clust4j.algo.BaseClassifierPlanner;
import com.clust4j.algo.preprocess.PreProcessor;

public abstract class Pipeline<T extends BaseClassifierPlanner> extends Clust4j {
	private static final long serialVersionUID = 3491192139356583621L;
	final transient Object fitLock = new Object();
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
	final static PreProcessor[] copyPipe(final PreProcessor... pipe) {
		final PreProcessor[] out = new PreProcessor[pipe.length];
		
		int idx = 0;
		for(PreProcessor pre: pipe)
			out[idx++] = pre.copy();
		
		return out;
	}
}
