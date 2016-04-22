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
package com.clust4j;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestGlobals {

	@Test
	public void testGammaFunctions() {
		// Test in the different ranges
		double val = 0.5;
		assertTrue(GlobalState.Mathematics.gamma(val) == 1.772453850905516);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 0.5723649429247001);
		
		val = 6.0;
		assertTrue(GlobalState.Mathematics.gamma(val) == 120);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 4.787491742782046);
		
		val = 12.1;
		assertTrue(GlobalState.Mathematics.gamma(val) == 5.098322784411637E7);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 17.747007270798743);
		
		val = 0.00001;
		assertTrue(GlobalState.Mathematics.gamma(val) == 1.0 / (val * (1.0 + GlobalState.Mathematics.GAMMA * val)));
		
		val = GlobalState.Mathematics.HIGH_BOUND + 1;
		assertTrue(GlobalState.Mathematics.gamma(val) == Double.POSITIVE_INFINITY);
		
		val = 0.0;
		boolean except = false;
		try { GlobalState.Mathematics.gamma(val);
		} catch(IllegalArgumentException i) { except = true;
		} finally { assertTrue(except); }
		
		except = false;
		try { GlobalState.Mathematics.lgamma(val);
		} catch(IllegalArgumentException i) { except = true;
		} finally { assertTrue(except); }
	}

}
