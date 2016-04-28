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

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.algo.BaseNeighborsModel;
import com.clust4j.algo.NearestNeighbors;
import com.clust4j.algo.NearestNeighborsParameters;
import com.clust4j.except.NaNException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class NearestNeighborImputation extends MatrixImputation {
	private static final long serialVersionUID = -2717555601058365204L;
	
	final static public int DEF_K = BaseNeighborsModel.DEF_K;
	final static public GeometricallySeparable DEF_METRIC = Distance.EUCLIDEAN;
	final static public CentralTendencyMethod DEF_CENT = CentralTendencyMethod.MEAN;
	
	private int k = DEF_K;
	private GeometricallySeparable sep = DEF_METRIC;
	private CentralTendencyMethod cent = DEF_CENT;
	
	
	
	
	public NearestNeighborImputation() {
		this(new NNImputationPlanner());
	}
	
	public NearestNeighborImputation(int k) {
		this(new NNImputationPlanner(k));
	}
	
	public NearestNeighborImputation(NNImputationPlanner planner) {
		super(planner);
		this.k = planner.k;
		this.cent = planner.cent;
		
		if(null == cent)
			throw new IllegalArgumentException("null method of central tendency");
		if(k < 1)
			throw new IllegalArgumentException("k must be greater than 0");
	}
	
	
	
	public static class NNImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private int k = DEF_K;
		private Random seed = new Random();
		private CentralTendencyMethod cent = DEF_CENT;
		
		public NNImputationPlanner() {}
		public NNImputationPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}

		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public NNImputationPlanner setK(final int k) {
			this.k = k;
			return this;
		}
		
		public NNImputationPlanner setMethodOfCentralTendency(final CentralTendencyMethod method) {
			this.cent = method;
			return this;
		}
		
		@Override
		public NNImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public NNImputationPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}
		
	}
	

	

	@Override
	public NearestNeighborImputation copy() {
		return new NearestNeighborImputation(new NNImputationPlanner()
			.setK(k)
			.setMethodOfCentralTendency(cent)
			.setSeed(getSeed())
			.setVerbose(verbose));
	}
	
	@Override
	public RealMatrix transform(final RealMatrix dat) {
		return new Array2DRowRealMatrix(transform(dat.getData()), false);
	}

	@Override
	public double[][] transform(final double[][] dat) {
		checkMat(dat);
		
		final LogTimer timer = new LogTimer();
		final int m = dat.length, n = dat[0].length, nc;
		final double[][] copy = MatUtils.copy(dat);
		
		final ArrayList<Integer> incompleteIndices = new ArrayList<>();
		final ArrayList<double[]> completeRecords = new ArrayList<>();
		
		
		// Get complete/non-complete matrices
		double[] row;
		info("separating complete from incomplete records");
		for(int i = 0; i < m; i++) {
			row = copy[i];
			if(VecUtils.containsNaN(row)) incompleteIndices.add(i);
			else completeRecords.add(row);
		}
		
		
		// Check k
		nc = completeRecords.size();
		String error;
		info(nc+" complete record" + (nc!=1?"s":"") + " extracted from input matrix");
		if(nc == 0) {
			error(new NaNException("no complete records in input matrix"));
		} else if(k > nc) {
			warn("number of complete records ("+nc
				+ ") is less than k ("+k+"); setting k to "+nc);
			k = nc;
		}
		
		
		// Build matrix
		final double[][] complete = MatUtils.fromList(completeRecords);
		final boolean mn = cent.equals(CentralTendencyMethod.MEAN);
		
		
		// Impute!
		info("imputing k nearest; method="+cent);
		int replacements;
		int[] nearest;
		NearestNeighbors nbrs;
		ArrayList<Integer> impute_indices;
		double[][] completeCols, nearestMat;
		double[] incomplete, completeRecord, col;
		for(Integer record: incompleteIndices) {
			incomplete = copy[record];
			impute_indices = new ArrayList<>(); // Hold the indices of columns which need to be imputed
			
			// Identify columns that need imputing
			for(int j = 0; j < n; j++)
				if(Double.isNaN(incomplete[j]))
					impute_indices.add(j);
			
			
			// Get complete cols
			replacements = impute_indices.size();
			if(replacements == n) {
				error = "record " + record + " is completely NaN";
				throw new NaNException(error);
			}
			
			completeRecord = exclude(incomplete, impute_indices);
			completeCols = excludeCols(complete, impute_indices);
			
			nbrs = new NearestNeighborsParameters(k)
					.setVerbose(false)
					.setSeed(getSeed())
					.setMetric(this.sep)
					.fitNewModel(new Array2DRowRealMatrix(completeCols, false)); // fits
			
			nearest = nbrs.getNeighbors(
				new Array2DRowRealMatrix(new double[][]{completeRecord}, 
					false)).getIndices()[0];
			
			
			nearestMat = MatUtils.getRows(complete, nearest);
			
			// Perform the imputation
			for(Integer imputationIdx: impute_indices) {
				col = MatUtils.getColumn(nearestMat, imputationIdx);
				incomplete[imputationIdx] = mn ? VecUtils.mean(col) : VecUtils.median(col);
			}
			
			info("record number "+record+" imputed in " + replacements + 
				" position" + (replacements!=1?"s":""));
		}
		
		sayBye(timer);
		return copy;
	}
	
	private static double[][] excludeCols(double[][] mat, ArrayList<Integer> exclude) {
		final int m = mat.length;
		final double[][] comp = new double[m][];
		
		for(int i = 0; i < m; i++)
			comp[i] = exclude(mat[i], exclude);
		
		return comp;
	}
	
	private static double[] exclude(double[] vec, ArrayList<Integer> exclude) {
		final double[] comp = new double[vec.length - exclude.size()];
		final int n = vec.length;
		
		int j = 0;
		for(int i = 0; i < n; i++) {
			if(exclude.contains(i))
				continue;
			comp[j++] = vec[i];
		}
		
		return comp;
	}
	
	
	public CentralTendencyMethod getCentralTendency() {
		return cent;
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.IMPUTE;
	}
	
	public int getK() {
		return k;
	}
	
	@Override
	public String getName() {
		return "NN imputation";
	}

	@Override final public NearestNeighborImputation fit(RealMatrix x){return this;}
}
