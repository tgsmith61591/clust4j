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
package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;
import com.clust4j.utils.Series.Inequality;

public class SeriesTests {

	@Test
	public void test() {
		final double a = 3.0, b = 5.0;
		assertTrue(Series.eval(a, Inequality.LESS_THAN, b));
		assertFalse(Series.eval(b, Inequality.LESS_THAN, a));
		
		assertTrue(Series.eval(a, Inequality.LESS_THAN_OR_EQUAL_TO, b));
		assertFalse(Series.eval(b, Inequality.LESS_THAN_OR_EQUAL_TO, a));

		assertTrue(Series.eval(b, Inequality.GREATER_THAN, a));
		assertFalse(Series.eval(a, Inequality.GREATER_THAN, b));

		assertTrue(Series.eval(b, Inequality.GREATER_THAN_OR_EQUAL_TO, a));
		assertFalse(Series.eval(a, Inequality.GREATER_THAN_OR_EQUAL_TO, b));

		assertTrue(Series.eval(a, Inequality.EQUAL_TO, a));
		assertFalse(Series.eval(a, Inequality.EQUAL_TO, b));
	}

}
