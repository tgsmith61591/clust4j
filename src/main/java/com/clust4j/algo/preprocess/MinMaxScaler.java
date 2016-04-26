package com.clust4j.algo.preprocess;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MinMaxScaler extends PreProcessor {
	private static final long serialVersionUID = 2028554388465841136L;
	public static final int DEF_MIN = 0;
	public static final int DEF_MAX = 1;
	
	volatile double[] mins;
	volatile double[] maxes;
	
	private final int min, max;
	
	private MinMaxScaler(MinMaxScaler instance) {
		this.mins = VecUtils.copy(instance.mins);
		this.maxes= VecUtils.copy(instance.maxes);
		this.min  = instance.min;
		this.max  = instance.max;
	}
	
	public MinMaxScaler() {
		this(DEF_MIN, DEF_MAX);
	}
	
	public MinMaxScaler(int min, int max) {
		if(min >= max)
			throw new IllegalStateException("RANGE_MIN ("+min+
					") must be lower than RANGE_MAX ("+max+")");
		
		this.min = min;
		this.max = max;
	}
	
	
	@Override
	protected void checkFit() {
		if(null == mins)
			throw new ModelNotFitException("model not yet fit");
	}
	
	@Override
	public MinMaxScaler copy() {
		return new MinMaxScaler(this);
	}

	@Override
	public MinMaxScaler fit(AbstractRealMatrix X) {
		synchronized(fitLock) {
			final int m = X.getRowDimension();
			final int n = X.getColumnDimension();
			
			this.mins = new double[n];
			this.maxes= new double[n];
			double[][] data = X.getData();
			
			for(int col = 0; col < n; col++) {
				double mn = Double.POSITIVE_INFINITY, mx = Double.NEGATIVE_INFINITY;
				
				for(int row = 0; row < m; row++) {
					mn = FastMath.min(mn, data[row][col]);
					mx = FastMath.max(mx, data[row][col]);
				}
				
				this.mins[col] = mn;
				this.maxes[col]= mx;
			}
			
			return this;
		}
	}

	@Override
	public AbstractRealMatrix transform(AbstractRealMatrix data) {
		return new Array2DRowRealMatrix(transform(data.getData()), false);
	}

	@Override
	public double[][] transform(double[][] data) {
		checkFit();
		MatUtils.checkDimsForUniformity(data);
		
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != mins.length)
			throw new DimensionMismatchException(n, mins.length);

		double[][] X = new double[m][n];
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			double mn = mins[j];
			double rng = maxes[j] - mn;
			
			for(int i = 0; i < m; i++) {
				X[i][j] = ((data[i][j] - mn) / rng) * (max - min) + min;
			}
		}
		
		// assign
		return X;
	}
}
