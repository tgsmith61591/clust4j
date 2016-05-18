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

import static org.junit.Assert.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

public class TestOptimizer {
	public final static double min_val = 0.8;
	final static OptimizableCaller caller = new  OptimizableCaller() {
		@Override
		public double doCall(double val) {
			final double a = 1.5;
			return FastMath.pow((val - a), 2) - min_val;
		}
	};

	@Test
	public void testBrentSimple() {
		BrentDownhillOptimizer optimizer= new BrentDownhillOptimizer(caller);
		double res = optimizer.optimize(); // optimize
		assertTrue(Precision.equals(res, 1.5, Precision.EPSILON));
		assertTrue(Precision.equals(optimizer.getFunctionResult(), -min_val, Precision.EPSILON));
		
		// if we try to "reoptimize" we get the same res:
		assertTrue(res == optimizer.optimize());
	}

	@Test
	public void testBrentWithBracket() {
		BrentDownhillOptimizer optimizer= new BrentDownhillOptimizer(caller, -3, -2);
		double res = optimizer.optimize(); // optimize
		assertTrue(Precision.equals(res, 1.5, Precision.EPSILON));
		assertTrue(optimizer.getNumFunctionCalls() == 5);
	}
	
	@Test
	public void testBadBracket() {
		boolean a = false;
		try {
			new BrentDownhillOptimizer(caller, 0, 0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
