package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.BaseNeighborsModel;
import com.clust4j.algo.NeighborsClassifierParameters;
import com.clust4j.algo.preprocess.PreProcessor;

public class NeighborsPipeline<M extends BaseNeighborsModel> 
		extends Pipeline<NeighborsClassifierParameters<M>> {

	private static final long serialVersionUID = 7363030699567515649L;

	public NeighborsPipeline(final NeighborsClassifierParameters<M> planner, final PreProcessor... pipe) {
		super(planner, pipe);
	}
	
	public M fit(final AbstractRealMatrix data) {
		synchronized(fitLock) {
			AbstractRealMatrix copy = data; // no need to copy, as handled in each processor...
			
			// Push through pipeline...
			for(PreProcessor pre: pipe)
				copy = pre.operate(copy);
	
			// Build the model -- the model should handle the dim check internally
			final M model = planner.fitNewModel(copy);
			
			// The model was fit internally
			return model;
		}
	}
}
