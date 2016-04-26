package com.clust4j.algo.preprocess;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.BaseModel;
import com.clust4j.algo.ModelSummary;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.Log;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.Axis;

public class PCA extends BaseModel implements PreProcessor, Loggable {
	private static final long serialVersionUID = 9041473302265494386L;
	final AbstractRealMatrix data;
	final int m, n;
	
	/*
	 * Run modes:
	 */
	private int n_components = -1;
	private double variability = Double.NaN;
	private boolean var_mode = false;
	
	/*
	 * Fit vars
	 */
	volatile private double[] means;
	volatile private double total_var = 0.0;
	volatile private double[] variabilities;
	volatile private double[] variability_ratio;
	volatile private RealMatrix components;
	volatile private double noise_variance;
	
	/*
	 * Whether there are warnings or not; shouldn't persist after saved
	 */
	volatile transient private boolean has_warnings = false;
	
	/**
	 * Copy constructor
	 * @param data
	 * @param n_components
	 * @param var
	 * @param vm
	 */
	private PCA(AbstractRealMatrix data, int n_components, double var, boolean vm) {
		// already mean centered
		this.data = (AbstractRealMatrix)data.copy();
		this.m = data.getRowDimension(); 
		this.n = data.getColumnDimension();
		this.n_components = n_components;
		this.variability = var;
		this.var_mode = vm;
	}
	
	/**
	 * Default constructor
	 * @param data
	 */
	private PCA(AbstractRealMatrix data) {
		this.m = data.getRowDimension();
		this.n = data.getColumnDimension();

		// need to mean center...
		double[][] X = new double[m][n], y = data.getData();
		this.means = new double[n];
		
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
		
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = y[i][j] - means[j];
			}
		}
		
		// assign
		this.data = new Array2DRowRealMatrix(X, false);
	}
	
	/**
	 * Construct an instance of PCA that will retain N components
	 * @param data
	 * @param n_components
	 */
	public PCA(AbstractRealMatrix data, int n_components) {
		this(data);
		if(n_components < 1 || n_components > n)
			error(new IllegalArgumentException("n_components must be "
				+ "greater than 0 and <= num cols in data"));
		
		this.n_components = n_components;
		logInit();
	}
	
	/**
	 * Construct an instance of PCA that will retain as many 
	 * components as explains the provided cumulative variability explained
	 * @param data
	 * @param variability_explained
	 */
	public PCA(AbstractRealMatrix data, double variability_explained) {
		this(data);
		if(variability_explained <= 0.0 || variability_explained > 1.0)
			error(new IllegalArgumentException("var_explained must be between 0 and 1.0"));
		
		this.variability = variability_explained;
		this.n_components = n;
		this.var_mode = true;
		logInit();
	}
	
	private void logInit() {
		ModelSummary ms = new ModelSummary(new Object[]{
			"n_components","var_thresh","var_mode"
		});
		
		ms.add(new Object[]{
			n_components, variability, var_mode
		});
		
		String[] fmt = formatter.format(ms).toString()
			.split(System.getProperty("line.separator"));
		for(String s : fmt)
			info(s);
	}
	
	/**
	 * Check if model is fit
	 */
	private void checkFit() {
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
	 * Get the cumulative sum of variability ratio explained by each component
	 * @return
	 */
	public double[] getVariabilityRatioExplained() {
		checkFit();
		return VecUtils.copy(this.variability_ratio);
	}
	
	
	
	/**
	 * Return a copy of the PCA model
	 * @return
	 */
	@Override
	public PCA copy() {
		PCA copy = new PCA(data, n_components, variability, var_mode);
		copy.total_var = this.total_var;
		copy.variabilities = VecUtils.copy(this.variabilities);
		copy.variability_ratio = VecUtils.copy(this.variability_ratio);
		copy.noise_variance = this.noise_variance;
		copy.components = null == this.components ? null : this.components.copy();
		copy.means = VecUtils.copy(means);
		
		return copy;
	}

	@Override
	public AbstractRealMatrix transform(AbstractRealMatrix data) {
		return new Array2DRowRealMatrix(transform(data.getData()), false);
	}

	@Override
	public double[][] transform(double[][] data) {
		checkFit();
		MatUtils.checkDimsForUniformity(data);
		final int n = data[0].length;
		
		// Check for dim equality
		if(n != this.means.length)
			error(new DimensionMismatchException(n, this.means.length));
		
		// Subtract the column means to center
		double[][] x = new double[data.length][n];
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < data.length; i++) {
				x[i][j] = data[i][j] - this.means[j];
			}
		}
		
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
	private static EntryPair<RealMatrix, RealMatrix> eigenSignFlip(RealMatrix U, RealMatrix V) {
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
	public PCA fit() {
		synchronized(fitLock) {
			if(null != components) // already fit
				return this;
			
			LogTimer timer = new LogTimer();
			
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
				
				info("need " + n_components + " component"+(n_components==1?"":"s")+
					" to explain " + (ratio_cumsum[n_components - 1]*100) + "% of variability");
			}
			
			
			// get noise variance
			if(n_components < FastMath.min(n, m)) {
				this.noise_variance = VecUtils.mean(VecUtils.slice(variabilities, n_components, s.length));
				info("noise variance: " + this.noise_variance);
			} else {
				this.noise_variance = 0.0;
			}
			
			
			// Set the components and other sliced variables
			this.components = new Array2DRowRealMatrix(MatUtils.slice(components_.getData(), 0, n_components), false);
			this.variabilities = VecUtils.slice(variabilities, 0, n_components);
			this.variability_ratio = VecUtils.slice(variability_ratio, 0, n_components);
			
			// log out
			info("cumulatively variability explained in retained "
				+ "component(s): " + ratio_cumsum[n_components - 1]);
			sayBye(timer);
			
			return this;
		}
	}

	@Override
	public void error(String msg) {
		Log.err(getLoggerTag(), msg);
	}

	@Override
	public void error(RuntimeException thrown) {
		error(thrown.getMessage());
		throw thrown;
	}

	@Override
	public void warn(String msg) {
		this.has_warnings = true;
		Log.warn(getLoggerTag(), msg);
	}

	@Override
	public void info(String msg) {
		Log.info(getLoggerTag(), msg);
	}

	@Override
	public void trace(String msg) {
		Log.trace(getLoggerTag(), msg);
	}

	@Override
	public void debug(String msg) {
		Log.debug(getLoggerTag(), msg);
	}

	@Override
	public void sayBye(LogTimer timer) {
		info("fit PCA model in " + timer.toString());
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.PRINCOMP;
	}

	@Override
	public boolean hasWarnings() {
		return has_warnings;
	}
}
