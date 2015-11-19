package com.clust4j.kernel;

import com.clust4j.utils.VecUtils;

/**
 * The Spline kernel is given as a piece-wise cubic polynomial, 
 * as derived in the works by <a href="http://www.svms.org/tutorials/Gunn1998.pdf">Gunn (1998)</a>.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class SplineKernel extends Kernel {
	
	public SplineKernel() {
		super();
	}

	@Override
	public double getSimilarity(double[] a, double[] b) {
		/*
		 * Kernlab's R package returns the following:
		 * 
		 * res <- 1 + x*y*(1+minv) - ((x+y)/2)*minv^2 + (minv^3)/3
         * fres <- prod(res)
         * 
         * 
         * We will split into three pieces:
         * 
         * fres <- prod(1 + front - mid + back)
		 */
		
		// Parallel min
		final double[] minV = VecUtils.pmin(a, b);
		
		// Get front
		final double[] front = VecUtils.multiply(VecUtils.multiply(a, b), VecUtils.scalarAdd(minV, 1d));
		
		// Get mid
		final double[] mid1 = VecUtils.scalarDivide(VecUtils.add(a, b), 2);
		final double[] mid2 = VecUtils.pow(minV, 2);
		final double[] mid = VecUtils.multiply(mid1, mid2);
		
		// Get back
		final double[] back = VecUtils.scalarDivide(VecUtils.pow(minV, 3), 3);
		
		// Calc res
		final double[] res = VecUtils.add(VecUtils.subtract(VecUtils.scalarAdd(front, 1), mid), back);
		return VecUtils.prod(res);
	}

	@Override
	public String getName() {
		return "SplineKernel";
	}

}
