/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.algo.preprocess.impute;

import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

/**
 * Imputes the missing values in a matrix with the column medians.
 * 
 * @author Taylor G Smith
 */
public class MedianImputation extends MatrixImputation {
	private static final long serialVersionUID = -3479934875830428010L;

	public MedianImputation() {
		this(new MedianImputationPlanner());
	}
	
	public MedianImputation(MedianImputationPlanner planner) {
		super(planner);
	}
	

	
	
	public static class MedianImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private Random seed = new Random();
		
		public MedianImputationPlanner() {}

		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public MedianImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public MedianImputationPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}
		
	}



	@Override
	public MedianImputation copy() {
		return new MedianImputation(new MedianImputationPlanner()
			.setSeed(getSeed())
			.setVerbose(verbose));
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.IMPUTE;
	}
	
	@Override
	public String getName() {
		return "Median imputation";
	}
	
	@Override
	public RealMatrix transform(final RealMatrix dat) {
		return new Array2DRowRealMatrix(transform(dat.getData()), false);
	}
	
	@Override
	public double[][] transform(final double[][] dat) {
		checkMat(dat);
		
		final LogTimer timer = new LogTimer();
		final double[][] copy = MatUtils.copy(dat);
		final int m = dat.length, n = dat[0].length;
		info("(" + getName() + ") performing median imputation on " + m + " x " + n + " dataset");
		
		// Operates in 2M * N
		for(int col = 0; col < n; col++) {
			final double median = VecUtils.nanMedian(MatUtils.getColumn(copy, col));

			int count = 0;
			for(int row = 0; row < m; row++) {
				if(Double.isNaN(copy[row][col])) {
					copy[row][col] = median;
					count++;
				}
			}
			
			info("(" + getName() + ") " + count + " NaN" + (count!=1?"s":"") + " identified in column " + col + " (column median="+median+")");
		}
		
		sayBye(timer);
		return copy;
	}

	@Override final public MedianImputation fit(RealMatrix x){return this;}
}
