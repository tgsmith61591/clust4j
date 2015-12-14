package com.clust4j.algo.prep;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.log.Log;
import com.clust4j.log.Loggable;
import com.clust4j.utils.NaNException;

public abstract class MatrixImputation implements Loggable {
	final public static boolean DEF_VERBOSE = AbstractClusterer.DEF_VERBOSE;
	protected final AbstractRealMatrix data;
	protected boolean verbose = DEF_VERBOSE;
	
	
	
	public MatrixImputation(final AbstractRealMatrix data) {
		this(data, true);
	}
	
	public MatrixImputation(final double[][] data) {
		this(new Array2DRowRealMatrix(data), false);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final boolean copy) {
		this.data = copy ? (AbstractRealMatrix)data.copy() : data;
		checkMat(data);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final ImputationPlanner planner) {
		this(data, true, planner);
	}
	
	public MatrixImputation(final double[][] data, final ImputationPlanner planner) {
		this(new Array2DRowRealMatrix(data), false, planner);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final boolean copy, final ImputationPlanner planner) {
		this(data, copy);
		this.verbose = planner.getVerbose();
	}
	
	
	
	abstract static public class ImputationPlanner {
		public ImputationPlanner(){}
		abstract public boolean getVerbose();
		abstract public ImputationPlanner setVerbose(boolean b);
	}
	
	
	
	
	
	final void checkMat(final AbstractRealMatrix data) {
		int m, n;
		if((m = data.getRowDimension()) < 1)
			throw new IllegalArgumentException("input data of length 0");
		if((n = data.getColumnDimension()) < 1)
			throw new IllegalArgumentException("input data with null column space");
		
		// Now check column NaN level
		final double[][] dataCopy = data.getData();
		for(int col = 0; col < n; col++) {
			Inner:
			for(int row = 0; row < m; row++) {
				if( !Double.isNaN(dataCopy[row][col]) )
					break Inner;
				else if(row == m - 1)
					throw new NaNException("column " + col + " is entirely NaN");
			}
		}
		
		// TODO?
		
		if(verbose) info("initializing matrix imputation method");
	}
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		Log.err(getLoggerTag(), msg);
	}
	
	@Override public void warn(String msg) {
		Log.warn(getLoggerTag(), msg);
	}
	
	@Override public void info(String msg) {
		Log.info(getLoggerTag(), msg);
	}
	
	@Override public void trace(String msg) {
		Log.trace(getLoggerTag(), msg);
	}
	
	@Override public void debug(String msg) {
		Log.debug(getLoggerTag(), msg);
	}
	
	abstract public double[][] impute();
}
