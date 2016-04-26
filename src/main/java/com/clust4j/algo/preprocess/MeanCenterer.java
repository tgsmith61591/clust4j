package com.clust4j.algo.preprocess;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MeanCenterer extends PreProcessor {
	private static final long serialVersionUID = 2028554388465841136L;
	volatile double[] means;
	
	private MeanCenterer(MeanCenterer instance) {
		this.means = VecUtils.copy(instance.means);
	}
	
	public MeanCenterer() {
	}
	
	
	@Override
	protected void checkFit() {
		if(null == means)
			throw new ModelNotFitException("model not yet fit");
	}
	
	@Override
	public MeanCenterer copy() {
		return new MeanCenterer(this);
	}

	@Override
	public MeanCenterer fit(AbstractRealMatrix data) {
		synchronized(fitLock) {
			final int m = data.getRowDimension();
			final int n = data.getColumnDimension();

			// need to mean center...
			this.means = new double[n];
			final double[][] y = data.getData();
			
			// First pass, compute mean...
			for(int j = 0; j < n; j++) {
				for(int i = 0; i < m; i++) {
					means[j] += y[i][j];
					
					// if last:
					if(i == m - 1) {
						means[j] /= (double)m;
					}
				}
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
		
		if(n != means.length)
			throw new DimensionMismatchException(n, means.length);

		double[][] X = new double[m][n];
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = data[i][j] - means[j];
			}
		}
		
		// assign
		return X;
	}

}
