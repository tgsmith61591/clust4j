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
package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

/**
 * The Spline kernel is given as a piece-wise cubic polynomial, 
 * as derived in the works by <a href="http://www.svms.org/tutorials/Gunn1998.pdf">Gunn (1998)</a>.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class SplineKernel extends Kernel {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5313152223880747371L;

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
		VecUtils.checkDims(a,b);
		final int n = a.length;
		final double[] minV = VecUtils.pmin(a, b);
		
		// Get front
		// Originally: 
		//
		// final double[] front = VecUtils.multiply(VecUtils.multiply(a, b), VecUtils.scalarAdd(minV, 1d));
		// final double[] mid1 = VecUtils.scalarDivide(VecUtils.add(a, b), 2);
		// final double[] mid2 = VecUtils.pow(minV, 2);
		// final double[] mid = VecUtils.multiplyForceSerial(mid1, mid2);
		// final double[] back = VecUtils.scalarDivide(VecUtils.pow(minV, 3), 3);
		// final double[] res = VecUtils.addForceSerial(VecUtils.subtractForceSerial(VecUtils.scalarAdd(front, 1), mid), back);
		// return VecUtils.prod(res);
		//
		// but this takes 12n (13n total!!)... can do it uglier, but much more elegantly in 1n (2n total):
		double[] front = new double[n], mid = new double[n], back = new double[n];
		double prod = 1;
		for(int i = 0; i < n; i++) {
			front[i] = a[i]*b[i] * (minV[i]+1);
			mid[i] = ((a[i]+b[i]) / 2) * (minV[i] * minV[i]);
			back[i] = FastMath.pow(minV[i], 3) / 3d;
			prod *= ( ((front[i]+1)-mid[i])+back[i] );
		}
		
		return prod;
	}

	@Override
	public String getName() {
		return "SplineKernel";
	}

}
