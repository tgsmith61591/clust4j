package com.clust4j.algo.prep;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.NearestNeighbors;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.NearestNeighbors.RunMode;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class NearestNeighborImputation extends MatrixImputation {
	final static public int DEF_K = NearestNeighbors.DEF_K;
	final static public double DEF_RADIUS = NearestNeighbors.DEF_EPS_RADIUS;
	final static public GeometricallySeparable DEF_METRIC = Distance.EUCLIDEAN;
	final static public RunMode DEF_RUN_MODE = RunMode.K_NEAREST;
	
	private int k = DEF_K;
	private double radius = DEF_RADIUS;
	private GeometricallySeparable sep = DEF_METRIC;
	private RunMode mode = DEF_RUN_MODE;
	private NearestNeighbors nnModel;
	
	

	public NearestNeighborImputation(AbstractRealMatrix data) {
		super(data);
		initModel();
	}
	
	public NearestNeighborImputation(final double[][] data) {
		super(data);
		initModel();
	}
	
	public NearestNeighborImputation(final AbstractRealMatrix data, final boolean copy) {
		super(data, copy);
		initModel();
	}
	
	public NearestNeighborImputation(AbstractRealMatrix data, NNImputationPlanner planner) {
		super(data, planner);
		initFromPlanner(planner);
		initModel();
	}
	
	public NearestNeighborImputation(final double[][] data, NNImputationPlanner planner) {
		super(data, planner);
		initFromPlanner(planner);
		initModel();
	}
	
	public NearestNeighborImputation(final AbstractRealMatrix data, final boolean copy, NNImputationPlanner planner) {
		super(data, copy, planner);
		initFromPlanner(planner);
		initModel();
	}
	
	
	
	public static class NNImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private int k = DEF_K;
		private double radius = DEF_RADIUS;
		private RunMode mode = DEF_RUN_MODE;
		
		public NNImputationPlanner() {}
		public NNImputationPlanner(int k) {
			this.k = k;
		}

		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public NNImputationPlanner setK(final int k) {
			this.k = k;
			return this;
		}
		
		public NNImputationPlanner setRadius(final double rad) {
			this.radius = rad;
			return this;
		}
		
		public NNImputationPlanner setRunMode(final RunMode mode) {
			this.mode = mode;
			return this;
		}

		@Override
		public NNImputationPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}
		
	}
	
	
	
	final void initFromPlanner(NNImputationPlanner planner) {
		this.k = planner.k;
		this.radius = planner.radius;
		this.mode = planner.mode;
	}
	
	final void initModel() {
		if(verbose) 
			info("fitting nearest neighbors model for " + 
				(mode.equals(RunMode.K_NEAREST)?("k="+k):("radius="+radius)));
		
		// TODO something here about the NaNs present so we fit on complete data...
		nnModel = new NearestNeighbors(data, 
			new NearestNeighborsPlanner()
				.setK(k)
				.setRadius(radius)
				.setSep(sep)
				.setRunMode(mode)).fit();
	}
	
	
	
	

	@Override
	public double[][] impute() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.IMPUTE;
	}
}
