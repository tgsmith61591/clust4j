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

package com.clust4j.algo.preprocess;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.Axis;

public class PCA extends Transformer {
	private static final long serialVersionUID = 9041473302265494386L;
	
	/*
	 * Run modes:
	 */
	private int n_components = -1;
	private double variability = Double.NaN;
	private boolean var_mode = false;
	
	/**
	 * Whether to retain U and S
	 */
	protected boolean retain = false;
	volatile protected RealMatrix S, U;
	
	/*
	 * Fit vars
	 */
	volatile int m, n;
	volatile MeanCenterer centerer;
	volatile private double total_var = 0.0;
	volatile private double[] variabilities;
	volatile protected double[] variability_ratio;
	volatile private double noise_variance;
	volatile protected RealMatrix components;
	
	/**
	 * Copy constructor
	 * @param data
	 * @param n_components
	 * @param var
	 * @param vm
	 */
	private PCA(PCA instance) {
		this.n_components = instance.n_components;
		this.variability = instance.variability;
		this.var_mode = instance.var_mode;
		
		this.m = instance.m;
		this.n = instance.n;
		this.centerer = null == instance.centerer ? null : instance.centerer.copy();
		this.total_var = instance.total_var;
		this.variabilities = VecUtils.copy(instance.variabilities);
		this.variability_ratio = VecUtils.copy(instance.variability_ratio);
		this.components = null == instance.components ? null : instance.components.copy();
		this.noise_variance = instance.noise_variance;
		this.S = null == instance.S ? null : instance.S.copy();
		this.U = null == instance.U ? null : instance.U.copy();
	}
	
	/**
	 * Construct an instance of PCA that will retain N components
	 * @param data
	 * @param n_components
	 */
	public PCA(int n_components) {
		if(n_components < 1)
			throw new IllegalArgumentException("n_components ("+n_components+") must be "
				+ "greater than 0");
		
		this.n_components = n_components;
	}
	
	/**
	 * Construct an instance of PCA that will retain as many 
	 * components as explains the provided cumulative variability explained
	 * @param data
	 * @param variability_explained
	 */
	public PCA(double variability_explained) {
		if(variability_explained <= 0.0 || variability_explained > 1.0)
			throw new IllegalArgumentException("var_explained must be between 0 and 1.0");
		
		this.variability = variability_explained;
		this.n_components = n;
		this.var_mode = true;
	}
	
	
	/**
	 * Check if model is fit
	 */
	@Override 
	protected void checkFit() {
		if(null == this.components)
			throw new ModelNotFitException("model not yet fit");
	}
	
	/**
	 * Return the components
	 * @return
	 */
	public RealMatrix getComponents() {
		checkFit();
		return this.components.copy();
	}
	
	/**
	 * Get the variability of the components not retained
	 * @return
	 */
	public double getNoiseVariance() {
		checkFit();
		return this.noise_variance;
	}
	
	/**
	 * Get the variability explained by each component
	 * @return
	 */
	public double[] getVariabilityExplained() {
		checkFit();
		return VecUtils.copy(this.variabilities);
	}
	
	/**
	 * Get the variability ratio explained by each component
	 * @return
	 */
	public double[] getVariabilityRatioExplained() {
		checkFit();
		return VecUtils.copy(this.variability_ratio);
	}
	
	/**
	 * Get the variability ratio explained by each component
	 * @return
	 */
	public double[] getCumulativeVariabilityRatioExplained() {
		checkFit();
		return VecUtils.cumsum(this.variability_ratio);
	}
	
	
	
	/**
	 * Return a copy of the PCA model
	 * @return
	 */
	@Override
	public PCA copy() {
		return new PCA(this);
	}

	@Override
	public RealMatrix transform(RealMatrix data) {
		return new Array2DRowRealMatrix(transform(data.getData()), false);
	}

	@Override
	public double[][] transform(double[][] data) {
		checkFit();
		MatUtils.checkDimsForUniformity(data);
		double[][] x = this.centerer.transform(data);
		
		// use block because it's faster for multiplication of potentially large matrices
		BlockRealMatrix X = new BlockRealMatrix(x);
		BlockRealMatrix transformed = X.multiply(this.components.transpose());
		
		return transformed.getData();
	}

	/**
	 * Flip Eigenvectors' sign to enforce deterministic output
	 * @param U
	 * @param V
	 * @return
	 */
	static EntryPair<RealMatrix, RealMatrix> eigenSignFlip(RealMatrix U, RealMatrix V) {
		// need to get column arg maxes of abs vals of U
		double[][] u = U.getData();
		double[][] v = V.getData();
		int[] max_abs_cols = MatUtils.argMax(MatUtils.abs(u), Axis.COL);
		
		// Get the signs of the diagonals in the rows corresponding to max_abs_cols
		int col_idx = 0;
		double val;
		double[] row;
		int[] signs = new int[U.getColumnDimension()];
		for(int row_idx: max_abs_cols) {
			row = u[row_idx];
			val = row[col_idx];
			signs[col_idx++] = val == 0 ? 0 : val < 0 ? -1 : 1;
		}
		
		// Multiply U by the signs... column-wise
		for(int i = 0; i < u.length; i++) {
			for(int j = 0; j < U.getColumnDimension(); j++) {
				u[i][j] *= signs[j];
			}
		}
		
		// Perform same op for V row-wise
		for(int j = 0; j < signs.length; j++) {
			for(int k = 0; k < V.getColumnDimension(); k++) {
				v[j][k] *= signs[j];
			}
		}
		
		return new EntryPair<RealMatrix, RealMatrix>(
			new Array2DRowRealMatrix(u, false),
			new Array2DRowRealMatrix(v, false)
		);
	}
	
	@Override
	public PCA fit(RealMatrix X) {
		synchronized(fitLock) {
			this.centerer = new MeanCenterer().fit(X);
			this.m = X.getRowDimension();
			this.n = X.getColumnDimension();
			
			// ensure n_components not too large
			if(this.n_components > n)
				this.n_components = n;
			
			final RealMatrix data = this.centerer.transform(X);
			SingularValueDecomposition svd = new SingularValueDecomposition(data);
			RealMatrix U = svd.getU(), S = svd.getS(), V = svd.getV().transpose();
			
			// flip Eigenvectors' sign to enforce deterministic output
			EntryPair<RealMatrix, RealMatrix> uv_sign_swap = eigenSignFlip(U, V);
			
			U = uv_sign_swap.getKey();
			V = uv_sign_swap.getValue();
			RealMatrix components_ = V;
			
			
			// get variance explained by singular value
			final double[] s = MatUtils.diagFromSquare(S.getData());
			this.variabilities = new double[s.length];
			for(int i= 0; i < s.length; i++) {
				variabilities[i] = (s[i]*s[i]) / (double)m;
				total_var += variabilities[i];
			}
			
			
			// get variability ratio
			this.variability_ratio = new double[s.length];
			for(int i = 0; i < s.length; i++) {
				variability_ratio[i] = variabilities[i] / total_var;
			}
			
			
			// post-process number of components if in var_mode
			double[] ratio_cumsum = VecUtils.cumsum(variability_ratio);
			if(this.var_mode) {
				for(int i = 0; i < ratio_cumsum.length; i++) {
					if(ratio_cumsum[i] >= this.variability) {
						this.n_components = i + 1;
						break;
					}
					
					// if it never hits the if block, the n_components is
					// equal to the number of columns in its entirety
				}
			}
			
			
			// get noise variance
			if(n_components < FastMath.min(n, m)) {
				this.noise_variance = VecUtils.mean(VecUtils.slice(variabilities, n_components, s.length));
			} else {
				this.noise_variance = 0.0;
			}
			
			
			// Set the components and other sliced variables
			this.components = new Array2DRowRealMatrix(MatUtils.slice(components_.getData(), 0, n_components), false);
			this.variabilities = VecUtils.slice(variabilities, 0, n_components);
			this.variability_ratio = VecUtils.slice(variability_ratio, 0, n_components);
			
			if(retain) {
				this.U = new Array2DRowRealMatrix(MatUtils.slice(U.getData(), 0, n_components), false);;
				this.S = new Array2DRowRealMatrix(MatUtils.slice(S.getData(), 0, n_components), false);;
			}
			
			
			return this;
		}
	}
	
	@Override
	public RealMatrix inverseTransform(RealMatrix X) {
		checkFit();
		
		// get the product of X times the components (not transposed)
		// and then add back in the mean...
		RealMatrix x = (RealMatrix) X.multiply(this.components);
		return this.centerer.inverseTransform(x);
	}
}
