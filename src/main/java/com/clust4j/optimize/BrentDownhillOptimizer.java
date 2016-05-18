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

public class BrentDownhillOptimizer extends BaseDownhillOptimizer {
	final static double mintol = 1.0e-11;
	final static double cg = 0.3819660;
	private int funcalls = 0;
	private int iter  = 0;
	private double xmin, fval;

	
	public BrentDownhillOptimizer(OptimizableCaller callable) {
		super(callable);
	}
	
	public BrentDownhillOptimizer(OptimizableCaller callable, double min, double max) {
		super(callable, min, max);
	}

	@Override
	protected double optimizeImplementation() {
		doCoreAlgorithm();
		return xmin;
	}
	
	private void doCoreAlgorithm() {
		double xa = bracket.xa;
		double xb = bracket.xb;
		double xc = bracket.xc;
		
		/*
		double fa = bracket.fa;
		double fb = bracket.fb;
		double fc = bracket.fc;
		*/
		
		funcalls  = bracket.funcalls;
		
		// begin core algo
		double x, w, v, fw, fv, fx, a, b;
		x = w = v = xb; // init all to xb
		fw= fv= fx= this.optimizer.doCall(x); // init all do f(x)
		
		if(xa < xc) {
			a = xa;
			b = xc;
		} else {
			a = xc;
			b = xa;
		}
		
		double deltAX = 0.0;
		funcalls = 1;
		iter = 0;
		
		double tol1, tol2, xmid, rat = 0.0, tmp1, tmp2, p, dxtmp, u, fu;
		while(iter < maxIter) {
			tol1 = tol * FastMath.abs(x) + mintol;
			tol2 = 2.0 * tol1;
			xmid = 0.5 * (a + b);
			
			// check for convergence
			if(FastMath.abs(x - xmid) < (tol2 - 0.5 * (b - a)))
				break;
			
			/*
			 * rat is only set in the true case of this. the first iteration
			 * should always be true, though, so initializing rat to 0.0 shouldn't
			 * cause any issues later...
			 */
			if(FastMath.abs(deltAX) <= tol1) {	// golden section step
				if(x >= xmid) {
					deltAX = a - x;	
				} else {
					deltAX = b - x;
				}
				
				rat = cg * deltAX;
			} else { 							// parabolic step
				tmp1 = (x - w) * (fx - fv);
				tmp2 = (x - v) * (fx - fw);
				p = (x - v) * tmp2 - (x - w) * tmp1;
				tmp2 = 2.0 * (tmp2 - tmp1);
				if(tmp2 > 0.0) {
					p = -p;
				}
				
				tmp2 = FastMath.abs(tmp2);
				dxtmp = deltAX;
				deltAX = rat;
				
				// check parabolic fit:
				if ((p > tmp2 * (a - x)) 
				&&  (p < tmp2 * (b - x)) 
				&&  (FastMath.abs(p) < FastMath.abs(0.5 * tmp2 * dxtmp))) {
					rat = p * 1.0 / tmp2;
					u = x + rat;
					
					if((u - a) < tol2 || (b - u) < tol2) {
						if(xmid - x >= 0) {
							rat = tol1;
						} else {
							rat = -tol1;
						}
					}
				} else {
					if(x >= xmid) {
						deltAX = a - x;
					} else {
						deltAX = b - x;
					}
					
					rat = cg * deltAX;
				}
			}
			
			
			// update by at least tol1
			if(FastMath.abs(rat) < tol1) {
				if(rat >= 0) {
					u = x + tol1;
				} else {
					u = x - tol1;
				}
			} else {
				u = x + rat;
			}
			
			fu = this.optimizer.doCall(u);
			funcalls++;
			
			
			// update values
			if(fu > fx) {
				if(u < x) {
					a = u;
				} else {
					b = u;
				}
				
				if(fu <= fw || w == x) {
					v = w;
					w = u;
					fv= fw;
					fw= fu;
				} else if((fu <= fv) || (v == x) || (v == w)) {
					v = u;
					fv= fu;
				}
			} else {
				if(u >= x) {
					a = x;
				} else {
					b = x;
				}
				
				v = w;
				w = x;
				x = u;
				fv= fw;
				fw= fx;
				fx= fu;
			}
			
			iter++;
		}
		
		// end core algorithm
		this.xmin = x;
		this.fval = fx;
		return;
	}

	@Override
	public int getNumFunctionCalls() {
		return funcalls;
	}

	@Override
	public double getFunctionResult() {
		return fval;
	}
}
