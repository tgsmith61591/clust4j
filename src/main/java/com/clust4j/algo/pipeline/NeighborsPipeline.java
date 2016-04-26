package com.clust4j.algo.pipeline;

import org.apache.commons.math3.linear.AbstractRealMatrix;

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
	
	public M fit(final AbstractRealMatrix data) {
		synchronized(fitLock) {
			AbstractRealMatrix copy = pipelineFitTransform(data);
	
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
