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

package com.clust4j.optimize;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;

/**
 * Build the functional bracket for the optimizing function.
 * Based on scipy's bracket optimization method.
 * @author Taylor G Smith
 * @see <a href="https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py">scipy</a>
 */
class Bracket {
	final static double growLimit = 110.0;
	final static int maxIter = 1000;
	
	final OptimizableCaller optimizer;
	protected double xa, xb, xc, fa, fb, fc;
	protected int funcalls;
	
	Bracket(OptimizableCaller optimizer, double xa, double xb) {
		this.optimizer = optimizer;
		this.xa = xa;
		this.xb = xb;
		
		// do core algorithm
		this.doCall();
	}
	
	private void doCall() {
		final double _gold = 1.618034;
		final double verysmall = GlobalState.Mathematics.EPS;
		final double twoverysmall = 2.0 * verysmall;
		
		// Get initial boundary values
		fa = optimizer.doCall(xa);
		fb = optimizer.doCall(xb);
		
		if(fa < fb) { // switch such that fa > fb
			double tmp;
			
			tmp = xa;
			xa = xb;
			xb = tmp;
			
			tmp = fa;
			fa = fb;
			fb = tmp;
		}
		
		// init xc and num calls
		xc = xb + _gold * (xb - xa);
		fc = optimizer.doCall(xc);
		funcalls = 3;
		
		// begin iterations
		int iter = 0;
		double tmp1, tmp2, val, denom, w, wlim, fw;
		while(fc < fb) {
			tmp1 = (xb - xa) * (fb - fc);
			tmp2 = (xb - xc) * (fb - fa);
			val = tmp2 - tmp1;
			
			if(FastMath.abs(val) < verysmall) {
				denom = twoverysmall;
			} else {
				denom = 2.0 * val;
			}
			
			w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
			wlim = xb + growLimit * (xc - xb);
			
			// check state of iter
			if(iter > maxIter)
				throw new RuntimeException("too many iterations: " + iter);
			
			iter++;
			if((w - xc) * (xb - w) > 0.0) {
				fw = optimizer.doCall(w);
				funcalls++;
				if(fw < fc) {
					xa = xb;
					xb = w;
					fa = fb;
					fb = fw;
					return;
				} else if(fw > fb) {
					xc = w;
					fc = fw;
					return;
				}
				
				w = xc + _gold * (xc - xb);
				fw = optimizer.doCall(w);
				funcalls++;
			} else if((w - wlim) * (wlim - xc) >= 0.0) {
				w = wlim;
				fw = optimizer.doCall(w);
				funcalls++;
			} else if((w - wlim) * (xc - w) > 0.0) {
				fw = optimizer.doCall(w);
				funcalls++;
				if(fw < fc) {
					xb = xc;
					xc = w;
					w = xc + _gold * (xc - xb);
					fb = fc;
					fc = fw;
					fw = optimizer.doCall(w);
					funcalls++;
				}
			} else {
				w = xc + _gold * (xc - xb);
				fw = optimizer.doCall(w);
				funcalls++;
			}
			
			// do reassignments
			xa = xb;
			xb = xc;
			xc = w;
	        fa = fb;
	        fb = fc;
	        fc = fw;
		}
		
		return;
	}
}
