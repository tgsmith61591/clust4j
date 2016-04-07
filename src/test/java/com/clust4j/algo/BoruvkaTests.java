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
package com.clust4j.algo;

import static org.junit.Assert.*;

import org.junit.Test;

public class BoruvkaTests {

	@Test
	public void testBallTreeMinDistDual() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		double rad1 = 0.5, rad2 = 0.75;
		assertTrue(BoruvkaAlgorithm.ballTreeMinDistDual(rad1, rad2, 1, 2, d) == 4.75);
	}

}
